// Copyright (C) 2026 REmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

use metal::{
    BinaryArchiveDescriptor, CompileOptions, Device, Function, Library,
    MeshRenderPipelineDescriptor, RenderPipelineDescriptor, RenderPipelineState, URL,
};
use remu_gpu::{ComputeDispatch, GpuRegister, IndexedDraw};
use remu_shader::{InstructionControl, Program, ShaderEvaluation, image_is_comparison};
use sha1::{Digest, Sha1};
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::mpsc::{TryRecvError, TrySendError};
use std::sync::{Arc, Mutex, OnceLock, RwLock};
use std::time::{Duration, Instant};

use super::{
    CompiledShaderPair, ComputePipelineKey, GraphicsPipelineKey, MetalPresenter,
    NGG_MESH_LDS_BYTES, NGG_MESH_THREADS, PIPELINE_COMPILE_COMPLETED, PIPELINE_COMPILE_ENQUEUED,
    PIPELINE_COMPILE_FAILED, PIPELINE_COMPILE_QUEUE_FULL, PersistentRenderPipelineArchive,
    ProgramIdentity, RenderPipelineArchiveState, ShaderCompileRequest, ShaderFunctionCache,
    ShaderFunctionCacheEntry, ShaderFunctionStageCache, diagnostic_capture_pixel_output,
    diagnostic_capture_vertex_position, diagnostic_disable_blending,
    diagnostic_dump_target_pixel_instruction_count, diagnostic_dump_target_pixel_shader,
    diagnostic_dump_video_chain_consumer, diagnostic_dump_video_draw,
    diagnostic_dump_video_target_consumer, diagnostic_force_msaa_pixel_color,
    diagnostic_force_msaa_vertex_pattern, diagnostic_force_vertex_fetch_id,
    diagnostic_force_visible_clip_depth, diagnostic_procedural_rectangle_fallback,
    diagnostic_video_chain_selected_draw, dump_msl_instruction_filter_matches,
    dump_msl_pixel_filter_matches, evaluation_samples_large_linear_video, image_component_class,
    metal_color_write_mask, pixel_input_controls, register_value, video_target_binding_summary,
};

impl MetalPresenter {
    pub(super) fn persist_render_pipeline_archive(&self) {
        if let Some(archive) = self.render_pipeline_archive.as_ref()
            && let Err(error) = archive.serialize()
        {
            eprintln!("[REMu][METAL] {error}");
        }
    }

    pub(super) fn compile_draw_shaders(&mut self, draw: &IndexedDraw, key: &GraphicsPipelineKey) {
        if self.compiled_shader_pairs.contains_key(key)
            || self.compiling_shader_pairs.contains(key)
            || self.failed_shader_pairs.contains(key)
        {
            return;
        }
        let request = ShaderCompileRequest {
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

    pub(super) fn receive_shader_compilations(&mut self) {
        loop {
            let result = match self.shader_compile_receiver.try_recv() {
                Ok(result) => result,
                Err(TryRecvError::Empty) => break,
                Err(TryRecvError::Disconnected) => {
                    for key in self.compiling_shader_pairs.drain() {
                        eprintln!(
                            "[REMu][METAL] Gen5 shader pair es={:#x} ps={:#x} failed: compiler worker stopped",
                            key.export_shader_address, key.pixel_shader_address
                        );
                        self.failed_shader_pairs.insert(key);
                    }
                    break;
                }
            };
            self.compiling_shader_pairs.remove(&result.key);
            match result.compiled {
                Ok(compiled) => {
                    PIPELINE_COMPILE_COMPLETED.fetch_add(1, Ordering::Relaxed);
                    eprintln!(
                        "[REMu][METAL] compiled Gen5 shader pair es={:#x} ps={:#x} stage={} MSL={vertex_bytes}/{pixel_bytes} bytes time={:.1}/{:.1}/{:.1}ms lower/function/pipeline reuse={vertex_reuse}/{pixel_reuse} diagnostic={vertex_pattern}/{pixel_color}",
                        result.key.export_shader_address,
                        result.key.pixel_shader_address,
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
                        vertex_pattern = compiled.vertex.source.contains("remu_diagnostic_vertex"),
                        pixel_color = compiled
                            .pixel
                            .source
                            .contains("float4(1.0f, 0.0f, 1.0f, 1.0f)"),
                    );
                    self.compiled_shader_pairs.insert(result.key, compiled);
                }
                Err(error) => {
                    self.failed_shader_pairs.insert(result.key.clone());
                    eprintln!(
                        "[REMu][METAL] Gen5 shader pair es={:#x} ps={:#x} failed: {error}",
                        result.key.export_shader_address, result.key.pixel_shader_address
                    );
                }
            }
        }
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
        let pixel = draw
            .pixel_shader
            .as_ref()
            .ok_or("draw has no decoded pixel shader")?;
        let export_evaluation = draw
            .export_shader_evaluation
            .as_ref()
            .ok_or("draw has no evaluated export shader resources")?;
        let pixel_evaluation = draw
            .pixel_shader_evaluation
            .as_ref()
            .ok_or("draw has no evaluated pixel shader resources")?;
        // Emit decoded IR before either stage is lowered. Keeping this ahead of `compile_msl`
        // makes the one-shot dump useful for precisely the shaders that fail translation instead
        // of only recording pairs that already compiled successfully.
        if let Some(directory) = std::env::var_os("REMU_DUMP_MSL_DIR")
            && dump_msl_pixel_filter_matches(draw.pixel_shader_address)
            && dump_msl_instruction_filter_matches(pixel.program.instructions.len())
            && let Err(error) = dump_shader_pair_ir(
                Path::new(&directory),
                draw.export_shader_address,
                draw.pixel_shader_address,
                &export.program.instructions,
                &pixel.program.instructions,
            )
        {
            eprintln!(
                "[REMu][METAL] could not dump shader IR for es={:#x} ps={:#x}: {error}",
                draw.export_shader_address, draw.pixel_shader_address
            );
        }
        if let Some(directory) = std::env::var_os("REMU_DUMP_GRAPHICS_LDS_IR_DIR")
            && (program_uses_lds(&export.program) || program_uses_lds(&pixel.program))
            && let Err(error) = dump_shader_pair_ir(
                Path::new(&directory),
                draw.export_shader_address,
                draw.pixel_shader_address,
                &export.program.instructions,
                &pixel.program.instructions,
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
            })
            .collect();
        let mut pixel_options = remu_shader_metal::MetalCompileOptions::pixel(
            pixel_outputs,
            register_value(&draw.context_registers, 0x1b3).unwrap_or_default(),
            register_value(&draw.context_registers, 0x1b4).unwrap_or_default(),
        );
        pixel_options.render_target_array_length = key.render_target_array_length;
        pixel_options.pixel_input_controls = key
            .pixel_input_controls
            .map(remu_shader_metal::PixelInputControl::from_register);
        let selected_diagnostic_shader = diagnostic_dump_target_pixel_shader()
            == Some(draw.pixel_shader_address)
            || diagnostic_dump_target_pixel_instruction_count()
                == Some(pixel.program.instructions.len())
            || (diagnostic_dump_video_target_consumer()
                && video_target_binding_summary(pixel_evaluation).is_some())
            || (diagnostic_dump_video_chain_consumer()
                && diagnostic_video_chain_selected_draw().load(Ordering::Relaxed) == draw.sequence);
        pixel_options.diagnostic_force_pixel_color = key.sample_count > 1
            && pixel.program.instructions.len() > 64
            && diagnostic_force_msaa_pixel_color();
        pixel_options.diagnostic_capture_pixel_output = diagnostic_capture_pixel_output()
            && ((key.sample_count > 1 && pixel.program.instructions.len() == 370)
                || selected_diagnostic_shader
                || (diagnostic_dump_video_draw()
                    && evaluation_samples_large_linear_video(pixel_evaluation)));
        let pixel_msl =
            remu_shader_metal::compile_msl(&pixel.program, pixel_evaluation, &pixel_options)
                .map_err(|error| format!("pixel lowering: {error}"))?;
        if key.render_target_array_length == 1 {
            let exported_parameters = vertex_parameter_export_mask(&export.program);
            let interpolated_attributes = pixel_interpolation_attribute_mask(&pixel.program);
            let missing = missing_vertex_parameter_exports(
                exported_parameters,
                interpolated_attributes,
                &key.pixel_input_controls,
            );
            if !missing.is_empty() {
                // NGG shaders may stage parameter exports through LDS/primitive export rather
                // than exposing a conventional EXP PARAM instruction in the decoded program.
                // Metal still requires the vertex and fragment interfaces to match exactly.
                // `required_vertex_output_count` below declares those physical locations and the
                // zero-initialized RemuVsOut supplies deterministic values until NGG staging is
                // lowered. Rejecting the pair here prevented even final scanout/composite draws
                // from executing and guaranteed a permanently black display. This matches the
                // other backend's established required-output behavior while keeping the missing
                // semantics explicit in the trace.
                eprintln!(
                    "[REMu][METAL][INTERFACE-FILL] draw={} indexed={} count={} instances={} primitive={:#x} target={:#x} es={:#x} exports={exported_parameters:#010x} ps={:#x} interpolates={interpolated_attributes:#010x} missing logical/physical/control={missing:08x?} vertex_inputs={:x?}",
                    draw.sequence,
                    draw.indexed,
                    draw.index_count,
                    draw.instance_count,
                    draw.primitive_type,
                    draw.color_target_address,
                    draw.export_shader_address,
                    draw.pixel_shader_address,
                    export_evaluation.vertex_inputs,
                );
            }
        }
        let uses_mesh_stage = draw.indexed
            && draw.instance_count == 1
            && draw.primitive_type == 4
            && ngg_mesh_export_stage(&export.program);
        let mut vertex_options = if uses_mesh_stage {
            remu_shader_metal::MetalCompileOptions::mesh(
                pixel_msl.interpolated_attribute_count,
                NGG_MESH_LDS_BYTES,
            )
        } else {
            remu_shader_metal::MetalCompileOptions::vertex(pixel_msl.interpolated_attribute_count)
        };
        vertex_options.render_target_array_length = key.render_target_array_length;
        vertex_options.diagnostic_force_visible_clip_depth =
            key.sample_count > 1 && diagnostic_force_visible_clip_depth();
        vertex_options.diagnostic_force_vertex_pattern =
            key.sample_count > 1 && diagnostic_force_msaa_vertex_pattern();
        vertex_options.diagnostic_force_vertex_fetch_id = diagnostic_force_vertex_fetch_id()
            && (key.sample_count > 1 || selected_diagnostic_shader);
        vertex_options.diagnostic_capture_vertex_position = diagnostic_capture_vertex_position()
            && (key.sample_count > 1
                || selected_diagnostic_shader
                || (diagnostic_dump_video_draw()
                    && evaluation_samples_large_linear_video(pixel_evaluation)));
        let vertex_msl = if uses_mesh_stage {
            remu_shader_metal::compile_msl(&export.program, export_evaluation, &vertex_options)
                .map_err(|error| format!("mesh lowering: {error}"))?
        } else if key.render_target_array_length > 1 {
            remu_shader_metal::compile_layered_rectangle_vertex(
                pixel_msl.interpolated_attribute_count,
                key.render_target_array_length,
            )
        } else if diagnostic_procedural_rectangle_fallback()
            && procedural_rectangle_export_stage(&export.program)
        {
            remu_shader_metal::compile_procedural_rectangle_vertex(
                pixel_msl.interpolated_attribute_count,
            )
        } else {
            remu_shader_metal::compile_msl(&export.program, export_evaluation, &vertex_options)
                .map_err(|error| format!("vertex lowering: {error}"))?
        };
        let lowering_duration = lowering_started.elapsed();
        if let Some(directory) = std::env::var_os("REMU_DUMP_MSL_DIR")
            && dump_msl_pixel_filter_matches(draw.pixel_shader_address)
            && dump_msl_instruction_filter_matches(pixel.program.instructions.len())
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
            descriptor.set_max_total_threads_per_mesh_threadgroup(NGG_MESH_THREADS);
            descriptor.set_mesh_threadgroup_size_is_multiple_of_thread_execution_width(true);
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
        })
    }
}
pub(super) const SHADER_COMPILE_QUEUE_CAPACITY: usize = 64;

pub(super) const SHADER_COMPILER_WORKER_COUNT: usize = 8;

pub(super) const METAL_LIBRARY_CACHE_FORMAT: &[u8] = b"remu-metal-library-v1\0";

pub(super) const METAL_RENDER_PIPELINE_ARCHIVE_FORMAT: &[u8] = b"remu-metal-render-pipeline-v1\0";

pub(super) const SLOW_PIPELINE_ARCHIVE_ADMISSION: Duration = Duration::from_millis(100);

pub(super) const LOADED_PIPELINE_ARCHIVE_CHECKPOINT_ADDITIONS: usize = 2;

impl PersistentRenderPipelineArchive {
    pub(super) fn open(device: &Device) -> Result<Self, String> {
        let path = metal_render_pipeline_archive_path(device);
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent).map_err(|error| {
                format!(
                    "could not create Metal pipeline cache {}: {error}",
                    parent.display()
                )
            })?;
        }
        // metal-rs 0.33 wraps +[NSURL URLWithString:] (an autoreleased object) as an owned URL.
        // Suppress the wrapper's release and let the active Objective-C pool perform the single
        // balanced release; otherwise archive serialization crashes at autoreleasePoolPop.
        let url = std::mem::ManuallyDrop::new(URL::new_with_string(&format!(
            "file://{}",
            path.display()
        )));
        let descriptor = BinaryArchiveDescriptor::new();
        let mut loaded = path.is_file();
        if loaded {
            descriptor.set_url(&url);
        }
        let archive = match device.new_binary_archive_with_descriptor(&descriptor) {
            Ok(archive) => archive,
            Err(error) if loaded => {
                eprintln!(
                    "[REMu][METAL] discarding unreadable pipeline archive {}: {error}",
                    path.display()
                );
                std::fs::remove_file(&path).map_err(|remove_error| {
                    format!(
                        "could not remove invalid pipeline archive {}: {remove_error}",
                        path.display()
                    )
                })?;
                loaded = false;
                device
                    .new_binary_archive_with_descriptor(&BinaryArchiveDescriptor::new())
                    .map_err(|error| format!("could not create Metal pipeline archive: {error}"))?
            }
            Err(error) => return Err(format!("could not create Metal pipeline archive: {error}")),
        };
        archive.set_label("REMu persistent render pipelines");
        eprintln!(
            "[REMu][METAL] {} render pipeline archive {}",
            if loaded { "loaded" } else { "created" },
            path.display()
        );
        Ok(Self {
            archive,
            path: Arc::new(path),
            loaded,
            state: Arc::new(Mutex::new(RenderPipelineArchiveState::default())),
            driver_lock: Arc::new(RwLock::new(())),
        })
    }

    pub(super) fn new_render_pipeline_state(
        &self,
        device: &Device,
        make_descriptor: impl FnOnce() -> Result<RenderPipelineDescriptor, String>,
    ) -> Result<RenderPipelineState, String> {
        if self.loaded {
            // Cached pipeline creation remains concurrent. Only a demonstrably slow archive miss
            // is admitted after creation, under the exclusive lock, so an old archive can learn
            // shaders from later scenes without serializing every ordinary cache hit.
            let descriptor = make_descriptor()?;
            descriptor.set_binary_archives(&[&self.archive]);
            let started = Instant::now();
            let pipeline = {
                let _reader = self
                    .driver_lock
                    .read()
                    .unwrap_or_else(std::sync::PoisonError::into_inner);
                device
                    .new_render_pipeline_state(&descriptor)
                    .map_err(|error| format!("Metal render pipeline: {error}"))?
            };
            if started.elapsed() >= SLOW_PIPELINE_ARCHIVE_ADMISSION {
                let _writer = self
                    .driver_lock
                    .write()
                    .unwrap_or_else(std::sync::PoisonError::into_inner);
                match self
                    .archive
                    .add_render_pipeline_functions_with_descriptor(&descriptor)
                {
                    Ok(true) => {
                        let should_serialize = {
                            let mut state = self
                                .state
                                .lock()
                                .unwrap_or_else(std::sync::PoisonError::into_inner);
                            state.additions = state.additions.saturating_add(1);
                            if !state.checkpoint_serialized
                                && state.additions >= LOADED_PIPELINE_ARCHIVE_CHECKPOINT_ADDITIONS
                            {
                                state.checkpoint_serialized = true;
                                true
                            } else {
                                false
                            }
                        };
                        if should_serialize && let Err(error) = self.serialize_locked() {
                            eprintln!(
                                "[REMu][METAL] slow-pipeline archive checkpoint failed: {error}"
                            );
                        }
                    }
                    Ok(false) => {}
                    Err(error) => eprintln!(
                        "[REMu][METAL] could not admit slow pipeline to loaded archive: {error}"
                    ),
                }
            }
            return Ok(pipeline);
        }
        // MTLBinaryArchive is documented as a resource shared by pipeline descriptors, but its
        // Rosetta driver path crashes while autoreleased descriptor internals from one thread are
        // drained concurrently with another archive mutation. Keep just descriptor/archive/
        // pipeline creation serialized and drain its local pool before admitting the next worker;
        // MSL lowering and metallib loading remain parallel outside this critical section.
        let _driver = self
            .driver_lock
            .write()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        objc::rc::autoreleasepool(|| {
            let descriptor = make_descriptor()?;
            descriptor.set_binary_archives(&[&self.archive]);
            self.archive
                .add_render_pipeline_functions_with_descriptor(&descriptor)
                .map_err(|error| {
                    format!("could not add Metal render pipeline to archive: {error}")
                })?;
            let should_serialize = {
                let mut state = self
                    .state
                    .lock()
                    .unwrap_or_else(std::sync::PoisonError::into_inner);
                state.additions = state.additions.saturating_add(1);
                if !state.checkpoint_serialized && state.additions >= 256 {
                    state.checkpoint_serialized = true;
                    true
                } else {
                    false
                }
            };
            let pipeline = device
                .new_render_pipeline_state(&descriptor)
                .map_err(|error| format!("Metal render pipeline: {error}"))?;
            // Diagnostic and interactive launches often end through SIGALRM or a window close
            // rather than a clean guest shutdown, so retain one mid-run checkpoint. The archive
            // can exceed 100 MiB for Unreal titles; rewriting it at every power of two additions
            // repeatedly stalls every compiler worker behind `driver_lock` during scene changes.
            // One checkpoint saves useful warm-up progress without turning a title transition
            // into nearly a gigabyte of synchronous archive writes. Clean shutdown still performs
            // the final serialization below.
            if should_serialize {
                self.serialize_locked()?;
            }
            Ok(pipeline)
        })
    }

    pub(super) fn serialize_locked(&self) -> Result<(), String> {
        let url = std::mem::ManuallyDrop::new(URL::new_with_string(&format!(
            "file://{}",
            self.path.display()
        )));
        self.archive
            .serialize_to_url(&url)
            .map(|_| ())
            .map_err(|error| {
                format!(
                    "could not persist Metal render pipeline archive {}: {error}",
                    self.path.display()
                )
            })
    }

    pub(super) fn serialize(&self) -> Result<(), String> {
        if self
            .state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .additions
            == 0
        {
            return Ok(());
        }
        let _driver = self
            .driver_lock
            .write()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        self.serialize_locked()
    }
}

impl ShaderFunctionCache {
    pub(super) fn function(
        cache: &ShaderFunctionStageCache,
        device: &Device,
        stage: &str,
        shader: &remu_shader_metal::MetalShader,
    ) -> Result<(Function, bool), String> {
        let mut entries = cache
            .entries
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        loop {
            match entries.get(&shader.source) {
                Some(ShaderFunctionCacheEntry::Ready(function)) => {
                    return Ok((function.clone(), true));
                }
                Some(ShaderFunctionCacheEntry::Failed(error)) => return Err(error.clone()),
                Some(ShaderFunctionCacheEntry::Compiling) => {
                    entries = cache
                        .completed
                        .wait(entries)
                        .unwrap_or_else(std::sync::PoisonError::into_inner);
                }
                None => {
                    entries.insert(shader.source.clone(), ShaderFunctionCacheEntry::Compiling);
                    break;
                }
            }
        }

        // Pipeline variants often reach different compiler workers together while sharing an
        // identical stage. Claim the exact source above before leaving the lock, so only one
        // worker invokes Apple's compiler or loads the persistent metallib. Other workers wait
        // for that result instead of compiling the same 100-400 KiB source concurrently.
        drop(entries);
        let compiled = (|| {
            let library = compile_metal_shader_library(device, stage, &shader.source)?;
            library
                .get_function(&shader.entry_point, None)
                .map_err(|error| format!("Metal {stage} function: {error}"))
        })();
        let mut entries = cache
            .entries
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        match &compiled {
            Ok(function) => {
                entries.insert(
                    shader.source.clone(),
                    ShaderFunctionCacheEntry::Ready(function.clone()),
                );
            }
            Err(error) => {
                entries.insert(
                    shader.source.clone(),
                    ShaderFunctionCacheEntry::Failed(error.clone()),
                );
            }
        }
        cache.completed.notify_all();
        compiled.map(|function| (function, false))
    }

    pub(super) fn vertex(
        &self,
        device: &Device,
        shader: &remu_shader_metal::MetalShader,
    ) -> Result<(Function, bool), String> {
        Self::function(&self.vertex, device, "vertex", shader)
    }

    pub(super) fn pixel(
        &self,
        device: &Device,
        shader: &remu_shader_metal::MetalShader,
    ) -> Result<(Function, bool), String> {
        Self::function(&self.pixel, device, "pixel", shader)
    }

    pub(super) fn mesh(
        &self,
        device: &Device,
        shader: &remu_shader_metal::MetalShader,
    ) -> Result<(Function, bool), String> {
        Self::function(&self.mesh, device, "mesh", shader)
    }
}

pub(super) fn compute_pipeline_key(
    dispatch: &ComputeDispatch,
    evaluation: &ShaderEvaluation,
    program: &std::sync::Arc<remu_shader::Program>,
) -> ComputePipelineKey {
    let image_slots = evaluation.image_binding_slots();
    ComputePipelineKey {
        // Retaining the Arc makes pointer identity stable for the lifetime of both successful and
        // negative cache entries; an allocator cannot recycle a dropped Program into this key.
        program_identity: ProgramIdentity(Arc::clone(program)),
        lds_size_bytes: dispatch.launch.lds_size_bytes,
        wave_lane_count: dispatch.launch.wave_lane_count,
        compute_system_registers: evaluation.compute_system_registers,
        initial_scalar_count: evaluation.initial_scalar_registers.len(),
        global_bindings: evaluation
            .global_memory_bindings
            .iter()
            .map(|binding| (binding.instruction_pcs.to_vec(), 0, 0, binding.writable))
            .collect(),
        image_bindings: evaluation
            .image_bindings
            .iter()
            .enumerate()
            .map(|(index, binding)| {
                let storage = is_storage_image_opcode(binding.opcode);
                let comparison = image_is_comparison(binding.opcode);
                let texture_type = binding
                    .resource_descriptor
                    .get(3)
                    .map_or(0, |word| (word >> 28) & 0xf);
                (
                    image_component_class(binding),
                    storage,
                    comparison,
                    binding.control.dimension,
                    texture_type,
                    image_slots.texture_slots[index],
                    image_slots.sampler_slots[index],
                )
            })
            .collect(),
    }
}

pub(super) fn program_embeds_global_base(program: &remu_shader::Program) -> bool {
    program.instructions.iter().any(|instruction| {
        matches!(
            instruction.control.as_ref(),
            Some(remu_shader::InstructionControl::GlobalMemory(_))
        )
    })
}

pub(super) fn is_writable_image_opcode(opcode: remu_shader::Opcode) -> bool {
    opcode.starts_with("ImageStore") || opcode.starts_with("ImageAtomic")
}

pub(super) fn is_storage_image_opcode(opcode: remu_shader::Opcode) -> bool {
    is_writable_image_opcode(opcode) || opcode.starts_with("ImageLoad")
}

pub(super) fn image_resource_descriptor_is_null(words: &[u32]) -> bool {
    !words.is_empty() && words.iter().all(|word| *word == 0)
}

pub(super) fn validate_compute_launch_for_metal(
    launch: remu_gpu::ComputeLaunch,
    execution_width: u32,
    wave64_split_safe: bool,
    maximum_threads_per_group: u64,
    maximum_threadgroup_memory_bytes: u64,
    static_threadgroup_memory_bytes: u64,
    dynamic_threadgroup_memory_bytes: u32,
) -> Result<(), String> {
    if launch.base_group != [0; 3] {
        return Err(format!(
            "Metal compute does not yet support nonzero base groups {:?}",
            launch.base_group
        ));
    }
    if execution_width != launch.wave_lane_count
        && !(launch.wave_lane_count == 64 && execution_width == 32 && wave64_split_safe)
    {
        return Err(format!(
            "guest wave{} cannot execute on Metal SIMD{} without changing wave semantics",
            launch.wave_lane_count, execution_width
        ));
    }
    let threadgroup_threads = launch.threads_per_group_total();
    if threadgroup_threads == 0 || threadgroup_threads > maximum_threads_per_group {
        return Err(format!(
            "guest compute threadgroup {:?} has {threadgroup_threads} threads, Metal permits {maximum_threads_per_group}",
            launch.threads_per_group
        ));
    }
    if dynamic_threadgroup_memory_bytes != 0
        && dynamic_threadgroup_memory_bytes != launch.lds_size_bytes
    {
        return Err(format!(
            "Metal shader requests {dynamic_threadgroup_memory_bytes} bytes of LDS, dispatch declares {}",
            launch.lds_size_bytes
        ));
    }
    let total_threadgroup_memory = static_threadgroup_memory_bytes
        .checked_add(u64::from(dynamic_threadgroup_memory_bytes))
        .ok_or("Metal threadgroup memory size overflows")?;
    if total_threadgroup_memory > maximum_threadgroup_memory_bytes {
        return Err(format!(
            "compute pipeline needs {total_threadgroup_memory} bytes of threadgroup memory, this Metal device permits {maximum_threadgroup_memory_bytes}"
        ));
    }
    Ok(())
}

pub(super) fn compute_program_can_split_wave64(program: &Program) -> bool {
    wave64_split_blockers(program).is_empty()
}

pub(super) fn wave64_split_blockers(program: &Program) -> Vec<String> {
    let mut blockers = Vec::new();
    for instruction in &program.instructions {
        let name = instruction.opcode.known_name();
        let wave_sensitive = name.is_none_or(|name| {
            ["Swizzle", "Ballot", "Saveexec"]
                .iter()
                .any(|wave_operation| name.contains(wave_operation))
                // Structured wave64 predication is represented half-locally across the two
                // Metal SIMD32 groups. These exercised operations consume only the current
                // architectural half of the mask, so each SIMD32 subgroup computes it exactly.
                && !matches!(name, "SAndSaveexecB64" | "SOrn2SaveexecB64")
        });
        if wave_sensitive {
            blockers.push(format!("{:#x}:{}", instruction.pc, instruction.opcode));
        }
    }
    blockers
}

pub(super) fn compile_metal_shader_library(
    device: &Device,
    stage: &str,
    source: &str,
) -> Result<Library, String> {
    if std::env::var_os("REMU_METAL_COMPILER").is_some() {
        return compile_metal_library_with_tools(device, stage, source);
    }

    let compile_options = CompileOptions::new();
    compile_options.set_fast_math_enabled(false);
    device
        .new_library_with_source(source, &compile_options)
        .map_err(|error| format!("Metal {stage} compilation: {error}"))
}

pub(super) fn compile_metal_compute_library(
    device: &Device,
    source: &str,
) -> Result<Library, String> {
    if std::env::var_os("REMU_METAL_COMPILER").is_some() {
        return compile_metal_library_with_tools(device, "compute", source);
    }
    let compile_options = CompileOptions::new();
    compile_options.set_fast_math_enabled(false);
    device
        .new_library_with_source(source, &compile_options)
        .map_err(|error| format!("Metal compute compilation: {error}"))
}

pub(super) fn compile_metal_library_with_tools(
    device: &Device,
    stage: &str,
    source: &str,
) -> Result<Library, String> {
    static NEXT_COMPILE_DIRECTORY: AtomicUsize = AtomicUsize::new(0);
    static CACHE_HITS: AtomicUsize = AtomicUsize::new(0);
    let compiler = std::env::var_os("REMU_METAL_COMPILER")
        .map(PathBuf::from)
        .ok_or("REMU_METAL_COMPILER is not configured")?;
    if !compiler.is_file() {
        return Err(format!("{} is not a file", compiler.display()));
    }
    let linker = compiler.with_file_name("metallib");
    if !linker.is_file() {
        return Err(format!("{} is not a file", linker.display()));
    }
    let cache_path = metal_library_cache_path(source);
    if cache_path.is_file() {
        match device.new_library_with_file(&cache_path) {
            Ok(library) => {
                let hit = CACHE_HITS.fetch_add(1, Ordering::Relaxed) + 1;
                if hit <= 8 || hit.is_power_of_two() {
                    eprintln!(
                        "[REMu][METAL] persistent shader cache hit={hit} stage={stage} path={}",
                        cache_path.display()
                    );
                }
                return Ok(library);
            }
            Err(error) => {
                PIPELINE_COMPILE_FAILED.fetch_add(1, Ordering::Relaxed);
                eprintln!(
                    "[REMu][METAL] discarding unreadable cached {stage} library {}: {error}",
                    cache_path.display()
                );
                let _ = std::fs::remove_file(&cache_path);
            }
        }
    }
    let sequence = NEXT_COMPILE_DIRECTORY.fetch_add(1, Ordering::Relaxed);
    let directory =
        std::env::temp_dir().join(format!("remu-metal-{}-{sequence}", std::process::id()));
    std::fs::create_dir(&directory)
        .map_err(|error| format!("could not create {}: {error}", directory.display()))?;
    let directory = MetalCompileDirectory(directory);
    let source_path = directory.0.join(format!("{stage}.metal"));
    let air_path = directory.0.join(format!("{stage}.air"));
    let library_path = directory.0.join(format!("{stage}.metallib"));
    std::fs::write(&source_path, source)
        .map_err(|error| format!("could not write {}: {error}", source_path.display()))?;
    run_metal_tool(
        Command::new(&compiler)
            .current_dir(&directory.0)
            .arg("-fmetal-math-mode=safe")
            .arg("-c")
            .arg(&source_path)
            .arg("-o")
            .arg(&air_path),
        &format!("{stage} compilation"),
    )?;
    run_metal_tool(
        Command::new(&linker)
            .current_dir(&directory.0)
            .arg(&air_path)
            .arg("-o")
            .arg(&library_path),
        &format!("{stage} library link"),
    )?;
    if let Some(parent) = cache_path.parent()
        && let Err(error) = std::fs::create_dir_all(parent)
    {
        eprintln!(
            "[REMu][METAL] could not create persistent shader cache {}: {error}",
            parent.display()
        );
    } else if let Err(error) = std::fs::rename(&library_path, &cache_path) {
        eprintln!(
            "[REMu][METAL] could not persist {stage} shader library {}: {error}",
            cache_path.display()
        );
    }
    let load_path = if cache_path.is_file() {
        &cache_path
    } else {
        &library_path
    };
    device
        .new_library_with_file(load_path)
        .map_err(|error| format!("could not load {}: {error}", load_path.display()))
}

pub(super) fn metal_library_cache_path(source: &str) -> PathBuf {
    let mut hasher = Sha1::new();
    hasher.update(METAL_LIBRARY_CACHE_FORMAT);
    hasher.update(source.as_bytes());
    let digest = format!("{:x}", hasher.finalize());
    metal_cache_root().join(format!("{digest}.metallib"))
}

pub(super) fn metal_cache_root() -> PathBuf {
    std::env::var_os("REMU_METAL_CACHE_DIR")
        .map(PathBuf::from)
        .or_else(|| {
            std::env::var_os("HOME")
                .map(PathBuf::from)
                .map(|home| home.join("Library/Caches/REMu/metal"))
        })
        .unwrap_or_else(|| std::env::temp_dir().join("remu-metal-cache"))
}

pub(super) fn metal_render_pipeline_archive_path(device: &Device) -> PathBuf {
    let mut hasher = Sha1::new();
    hasher.update(METAL_RENDER_PIPELINE_ARCHIVE_FORMAT);
    hasher.update(device.name().as_bytes());
    hasher.update(std::env::consts::ARCH.as_bytes());
    let digest = format!("{:x}", hasher.finalize());
    metal_cache_root().join(format!("render-{digest}.metallibarchive"))
}

pub(super) fn run_metal_tool(command: &mut Command, operation: &str) -> Result<(), String> {
    let output = command
        .output()
        .map_err(|error| format!("could not start {operation}: {error}"))?;
    if output.status.success() {
        return Ok(());
    }
    let diagnostics = String::from_utf8_lossy(&output.stderr);
    Err(format!(
        "{operation} exited with {}: {}",
        output.status,
        diagnostics.trim()
    ))
}

pub(super) struct MetalCompileDirectory(PathBuf);

impl Drop for MetalCompileDirectory {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

pub(super) fn shader_scalar_metadata(evaluation: &ShaderEvaluation) -> Result<Vec<u32>, String> {
    let mut words = evaluation.initial_scalar_registers.to_vec();
    words.reserve(evaluation.global_memory_bindings.len().saturating_mul(4));
    for binding in &evaluation.global_memory_bindings {
        words.extend([
            binding.base_address as u32,
            (binding.base_address >> 32) as u32,
            u32::try_from(binding.byte_count)
                .map_err(|_| "shader global binding exceeds the 32-bit Metal byte range")?,
            u32::from(binding.writable),
        ]);
    }
    Ok(words)
}

pub(super) fn shader_scalar_metadata_for_registers(
    evaluation: &ShaderEvaluation,
    shader_registers: &[GpuRegister],
    user_data_register: u32,
    scalar_register_base: usize,
) -> Result<Vec<u32>, String> {
    debug_assert!(
        shader_registers
            .windows(2)
            .all(|registers| registers[0].offset <= registers[1].offset)
    );
    let mut words = shader_scalar_metadata(evaluation)?;
    let scalar_end = scalar_register_base
        .saturating_add(64)
        .min(evaluation.initial_scalar_registers.len());
    for scalar_register in scalar_register_base..scalar_end {
        let offset = u32::try_from(scalar_register - scalar_register_base)
            .map_err(|_| "shader scalar register offset does not fit u32")?;
        words[scalar_register] =
            register_value(shader_registers, user_data_register + offset).unwrap_or_default();
    }
    Ok(words)
}

pub(super) fn program_uses_lds(program: &remu_shader::Program) -> bool {
    program.instructions.iter().any(|instruction| {
        instruction.encoding == remu_shader::ShaderEncoding::Ds
            && !instruction.opcode.is("DsSwizzleB32")
    })
}

pub(super) fn dump_shader_pair_msl(
    directory: &Path,
    export_address: u64,
    pixel_address: u64,
    vertex_source: &str,
    pixel_source: &str,
) -> Result<(), String> {
    std::fs::create_dir_all(directory)
        .map_err(|error| format!("could not create {}: {error}", directory.display()))?;
    let stem = format!("es-{export_address:x}-ps-{pixel_address:x}");
    let vertex_path = directory.join(format!("{stem}-vertex.metal"));
    let pixel_path = directory.join(format!("{stem}-pixel.metal"));
    std::fs::write(&vertex_path, vertex_source)
        .map_err(|error| format!("could not write {}: {error}", vertex_path.display()))?;
    std::fs::write(&pixel_path, pixel_source)
        .map_err(|error| format!("could not write {}: {error}", pixel_path.display()))?;
    Ok(())
}

pub(super) fn dump_shader_pair_ir(
    directory: &Path,
    export_address: u64,
    pixel_address: u64,
    export: &[remu_shader::Instruction],
    pixel: &[remu_shader::Instruction],
) -> Result<(), String> {
    std::fs::create_dir_all(directory)
        .map_err(|error| format!("could not create {}: {error}", directory.display()))?;
    let stem = format!("es-{export_address:x}-ps-{pixel_address:x}");
    let path = directory.join(format!("{stem}-ir.txt"));
    let contents = format!("export={export:#?}\n\npixel={pixel:#?}\n");
    std::fs::write(&path, contents)
        .map_err(|error| format!("could not write {}: {error}", path.display()))
}

pub(super) fn dump_compute_shader_msl(
    directory: &Path,
    compute_address: u64,
    instructions: &[remu_shader::Instruction],
    source: &str,
) -> Result<(), String> {
    std::fs::create_dir_all(directory)
        .map_err(|error| format!("could not create {}: {error}", directory.display()))?;
    let source_path = directory.join(format!("cs-{compute_address:x}.metal"));
    std::fs::write(&source_path, source)
        .map_err(|error| format!("could not write {}: {error}", source_path.display()))?;
    let ir_path = directory.join(format!("cs-{compute_address:x}-ir.txt"));
    std::fs::write(&ir_path, format!("{instructions:#?}\n"))
        .map_err(|error| format!("could not write {}: {error}", ir_path.display()))
}

pub(super) fn vertex_parameter_export_mask(program: &Program) -> u32 {
    program
        .instructions
        .iter()
        .filter_map(|instruction| {
            let InstructionControl::Export(export) = instruction.control.as_ref()? else {
                return None;
            };
            ((32..64).contains(&export.target) && export.enable_mask != 0)
                .then_some(1_u32 << (export.target - 32))
        })
        .fold(0, |mask, export| mask | export)
}

pub(super) fn pixel_interpolation_attribute_mask(program: &Program) -> u32 {
    program
        .instructions
        .iter()
        .filter_map(|instruction| {
            let InstructionControl::Interpolation(interpolation) = instruction.control.as_ref()?
            else {
                return None;
            };
            (interpolation.attribute < 32).then_some(1_u32 << interpolation.attribute)
        })
        .fold(0, |mask, attribute| mask | attribute)
}

pub(super) fn missing_vertex_parameter_exports(
    exported_parameters: u32,
    interpolated_attributes: u32,
    controls: &[u32; 32],
) -> Vec<(u32, u32, u32)> {
    (0..32)
        .filter(|logical| interpolated_attributes & (1 << logical) != 0)
        .filter_map(|logical| {
            let raw = controls[logical as usize];
            let control = remu_shader_metal::PixelInputControl::from_register(raw);
            if control.uses_default() {
                return None;
            }
            let physical = u32::from(control.input_offset);
            (exported_parameters & (1 << physical) == 0).then_some((logical, physical, raw))
        })
        .collect()
}

pub(super) fn skip_incomplete_rectangle_draw(draw: &IndexedDraw) -> bool {
    // A non-default pixel interpolation has no defined source when the vertex/primitive stage
    // exports no matching PARAM. Synthesizing zero varyings turns the trailing invalid RECTLIST
    // packets in Silent Hill into full-screen black clears, erasing the correctly rendered
    // composite immediately before every flip. Treat the invalid interface as a rejected draw,
    // which preserves the last valid target contents. Keep an explicit diagnostic escape hatch
    // for comparing the old zero-fill behavior.
    static ALLOW_INVALID: OnceLock<bool> = OnceLock::new();
    !*ALLOW_INVALID.get_or_init(|| {
        std::env::var_os("REMU_ALLOW_INCOMPLETE_RECTANGLES").is_some_and(|value| value == "1")
    }) && rectangle_draw_has_incomplete_interface(draw)
        && !(diagnostic_procedural_rectangle_fallback()
            && draw
                .export_shader
                .as_ref()
                .is_some_and(|shader| procedural_rectangle_export_stage(&shader.program)))
}

/// Recognizes a fused GFX10 NGG export/geometry program that requires a Metal mesh stage.
///
/// These programs allocate primitive output, exchange records through general LDS, then publish
/// both primitive connectivity and position. Lane-private ADDTID compiler spills alone do not
/// qualify: ordinary vertex/pixel lowering already preserves those without a mesh pipeline.
pub(super) fn ngg_mesh_export_stage(program: &Program) -> bool {
    let mut allocation_message = false;
    let mut primitive_export = false;
    let mut position_export = false;
    let mut general_lds = false;
    for instruction in &program.instructions {
        if instruction.opcode.is("SSendmsg")
            && instruction
                .sources
                .first()
                .is_some_and(|source| source.value == 9)
        {
            allocation_message = true;
        }
        if let Some(InstructionControl::Export(export)) = instruction.control.as_ref() {
            primitive_export |= export.target == 20 && export.enable_mask != 0;
            position_export |= export.target == 12 && export.enable_mask != 0;
        }
        if matches!(instruction.control, Some(InstructionControl::DataShare(_)))
            && !matches!(
                instruction.opcode.known_name(),
                Some("DsSwizzleB32" | "DsWriteAddtidB32" | "DsReadAddtidB32")
            )
        {
            general_lds = true;
        }
    }
    allocation_message && primitive_export && position_export && general_lds
}

/// Recognizes the position-only AGC primitive stage used for procedural fullscreen rectangles.
///
/// This deliberately describes architectural behavior instead of matching a game address or an
/// exact byte string: the stage sends GS allocation message 9, exports primitive data at target
/// 20, derives X/Y from the vertex ID, and exports position at target 12. It must have no memory,
/// LDS, image, interpolation, or PARAM-export work. More complex merged ES/GS programs are not
/// accepted here and continue through the ordinary translator/rejection path.
pub(super) fn procedural_rectangle_export_stage(program: &Program) -> bool {
    let mut primitive_export = false;
    let mut position_export = false;
    let mut allocation_message = false;
    let mut vertex_id_x = false;
    let mut vertex_id_y = false;
    let mut coordinate_conversions = 0_u32;
    let mut coordinate_mads = 0_u32;

    for instruction in &program.instructions {
        match instruction.control.as_ref() {
            Some(InstructionControl::Export(export)) => match export.target {
                20 if export.enable_mask != 0 && export.done => primitive_export = true,
                12 if export.enable_mask == 0x0f && export.done => position_export = true,
                _ => return false,
            },
            Some(
                InstructionControl::Image(_)
                | InstructionControl::GlobalMemory(_)
                | InstructionControl::BufferMemory(_)
                | InstructionControl::Interpolation(_)
                | InstructionControl::DataShare(_),
            ) => return false,
            _ => {}
        }

        match instruction.opcode.known_name() {
            Some("SSendmsg")
                if instruction
                    .sources
                    .first()
                    .is_some_and(|source| source.value == 9) =>
            {
                allocation_message = true;
            }
            Some("VAndB32") => vertex_id_x = true,
            Some("VLshrrevB32") => vertex_id_y = true,
            Some("VCvtF32U32" | "VCvtF32I32") => {
                coordinate_conversions = coordinate_conversions.saturating_add(1);
            }
            Some("VMadF32" | "VFmaF32" | "VLshlAddU32") => {
                coordinate_mads = coordinate_mads.saturating_add(1);
            }
            Some(name)
                if name.starts_with("Ds")
                    || name.starts_with("Image")
                    || name.starts_with("Buffer")
                    || name.starts_with("Global")
                    || name.starts_with("Flat") =>
            {
                return false;
            }
            None => return false,
            _ => {}
        }
    }

    primitive_export
        && position_export
        && allocation_message
        && vertex_id_x
        && vertex_id_y
        && coordinate_conversions >= 2
        && coordinate_mads >= 2
}

pub(super) fn rectangle_draw_has_incomplete_interface(draw: &IndexedDraw) -> bool {
    if !matches!(draw.primitive_type, 7 | 0x11) || draw.indexed || draw.index_count != 3 {
        return false;
    }
    let Some(export) = draw.export_shader.as_ref() else {
        return false;
    };
    let Some(pixel) = draw.pixel_shader.as_ref() else {
        return false;
    };
    let exported_parameters = vertex_parameter_export_mask(&export.program);
    let interpolated_attributes = pixel_interpolation_attribute_mask(&pixel.program);
    !missing_vertex_parameter_exports(
        exported_parameters,
        interpolated_attributes,
        &pixel_input_controls(&draw.context_registers),
    )
    .is_empty()
}

pub(super) fn pixel_scalars(draw: &IndexedDraw) -> [u32; 4] {
    std::array::from_fn(|index| {
        register_value(
            &draw.shader_registers,
            0x00c + u32::try_from(index).expect("scalar index fits"),
        )
        .unwrap_or_default()
    })
}
