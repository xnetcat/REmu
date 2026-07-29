// Copyright (C) 2026 REmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

use metal::{Device, MeshRenderPipelineDescriptor, RenderPipelineDescriptor};
use remu_gpu::IndexedDraw;
use std::path::Path;
use std::sync::atomic::Ordering;
use std::sync::mpsc::{TryRecvError, TrySendError};
use std::time::Instant;

use super::{
    CompiledShaderPair, ComputePipelineKey, GraphicsPipelineKey, MetalPresenter,
    NGG_MESH_LDS_BYTES, NGG_MESH_THREADS, PIPELINE_COMPILE_COMPLETED, PIPELINE_COMPILE_ENQUEUED,
    PIPELINE_COMPILE_FAILED, PIPELINE_COMPILE_QUEUE_FULL, PersistentRenderPipelineArchive,
    ProgramIdentity, RenderPipelineArchiveState, ShaderCompileRequest, ShaderFunctionCache,
    ShaderFunctionCacheEntry, ShaderFunctionStageCache, diagnostic_capture_pixel_output,
    diagnostic_capture_vertex_position, diagnostic_disable_blending,
    diagnostic_dump_shadow_consumer, diagnostic_dump_target_pixel_instruction_count,
    diagnostic_dump_target_pixel_shader, diagnostic_dump_video_chain_consumer,
    diagnostic_dump_video_draw, diagnostic_dump_video_target_consumer,
    diagnostic_force_msaa_pixel_color, diagnostic_force_msaa_vertex_pattern,
    diagnostic_force_vertex_fetch_id, diagnostic_force_visible_clip_depth,
    diagnostic_video_chain_selected_draw, dump_msl_instruction_filter_matches,
    dump_msl_pixel_filter_matches, evaluation_samples_large_linear_video,
    evaluation_samples_layered_comparison, image_component_class, metal_color_write_mask,
    register_value, video_target_binding_summary,
};
mod archive;
mod compile;
mod program;
pub(crate) use self::compile::{
    compile_metal_compute_library, compile_metal_library_with_tools, compile_metal_shader_library,
    dump_compute_shader_msl, dump_shader_pair_ir, dump_shader_pair_msl, metal_cache_root,
};
#[cfg(test)]
pub(crate) use self::program::{
    rectangle_draw_has_incomplete_interface, shader_scalar_metadata,
};
pub(crate) use self::program::{
    compute_pipeline_key, compute_program_can_split_wave64,
    graphics_allocated_vector_register_count, image_resource_descriptor_is_null,
    is_storage_image_opcode, is_writable_image_opcode, missing_vertex_parameter_exports,
    ngg_mesh_export_stage, pixel_interpolation_attribute_mask, pixel_scalars,
    procedural_rectangle_export_stage, program_embeds_global_base, program_uses_lds,
    rectangle_coordinate_parameter_mask, shader_scalar_metadata_for_registers,
    validate_compute_launch_for_metal, vertex_parameter_export_mask, wave64_split_blockers,
};

impl MetalPresenter {
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
        pixel_options.allocated_vector_register_count = key.pixel_allocated_vector_register_count;
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
                && diagnostic_video_chain_selected_draw().load(Ordering::Relaxed) == draw.sequence)
            || (diagnostic_dump_shadow_consumer()
                && evaluation_samples_layered_comparison(pixel_evaluation));
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
        let uses_procedural_rectangle_stage = !draw.indexed
            && draw.index_count == 3
            && matches!(draw.primitive_type, 7 | 0x11)
            && procedural_rectangle_export_stage(&export.program);
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
        vertex_options.allocated_vector_register_count = key.export_allocated_vector_register_count;
        vertex_options.render_target_array_length = key.render_target_array_length;
        vertex_options.vertex_output_control = key.vertex_output_control;
        vertex_options.diagnostic_force_visible_clip_depth =
            key.sample_count > 1 && diagnostic_force_visible_clip_depth();
        vertex_options.diagnostic_force_vertex_pattern = diagnostic_force_msaa_vertex_pattern()
            && (key.sample_count > 1 || selected_diagnostic_shader);
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
        } else if uses_procedural_rectangle_stage {
            let coordinate_parameter_mask =
                rectangle_coordinate_parameter_mask(&pixel.program, &key.pixel_input_controls)
                    .unwrap_or_else(|| {
                        // A sample-free procedural consumer has no data-flow evidence for
                        // parameter roles. Preserve the previous convention for that unrelated
                        // case: one input is UV, while input zero of a wider interface is tint.
                        if pixel_msl.interpolated_attribute_count == 1 {
                            1
                        } else {
                            u32::MAX & !1
                        }
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
