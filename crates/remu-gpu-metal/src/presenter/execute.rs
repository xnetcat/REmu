// Copyright (C) 2026 REmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

use super::WrittenRegion;
use super::{
    CB_BLEND_ALPHA, CB_BLEND_BLUE, CB_BLEND_GREEN, CB_BLEND_RED, CB_COLOR_CONTROL,
    CB_COLOR_REGISTER_STRIDE, CB_COLOR0_FMASK, CB_COLOR0_FMASK_BASE_EXT, ColorOperationMode,
    DB_DEPTH_CONTROL, DB_DEPTH_SIZE_XY, DB_Z_INFO, DEFAULT_CB_COLOR_CONTROL,
    DEFAULT_PA_SU_SC_MODE_CNTL, DIAGNOSTIC_MESH_POSITION_BUFFER_INDEX,
    DIAGNOSTIC_PIXEL_OUTPUT_BUFFER_INDEX, DIAGNOSTIC_PIXEL_OUTPUT_HEIGHT,
    DIAGNOSTIC_PIXEL_OUTPUT_WIDTH, DIAGNOSTIC_PIXEL_OUTPUT_WORDS_PER_PIXEL,
    DIAGNOSTIC_VERTEX_POSITION_BUFFER_INDEX,
    DIAGNOSTIC_VERTEX_POSITION_VERTEX_CAPACITY, DIAGNOSTIC_VERTEX_STATE_WORDS_PER_VERTEX,
    DiagnosticScanoutWriter, FlushPolicy, GUEST_BATCH_WORK_CAPACITY, GraphicsPipelineKey,
    GuestBatchFlushReason, GuestBatchWork, GuestRenderPassKey, GuestScissor,
    GuestTextureDescriptor, MetalPresenter, NGG_MESH_DRAW_BUFFER_INDEX, NGG_MESHLET_BUFFER_INDEX,
    OpenGuestRenderPass, PA_SC_AA_MASK_X0Y0_X1Y0, PA_SC_AA_MASK_X0Y1_X1Y1, PA_SU_SC_MODE_CNTL,
    ResidencyState, ResidentTextureActivityStamp, SharedEventDrawTextureOverride,
    SharedEventWriteKind, SpanQueryOrder, StaleSource, append_unique_global_writebacks,
    append_unique_guest_write_ranges, canonical_texture_bindings, capture_guest_frame,
    capture_target_sample_textures, collect_pending_global_writebacks,
    collect_pending_guest_write_ranges, collect_shared_event_global_versions,
    color_operation_is_raster_noop, compute_pipeline_key, compute_program_can_split_wave64,
    consume_color_metadata_clear, consume_depth_metadata_clear, create_global_argument_buffer,
    decode_color_eliminate_fast_clear_target, decode_color_operation_mode,
    decode_color_resolve_targets, decode_depth_target, decode_image_descriptor,
    decode_raster_state, decode_render_targets, decode_scissor, decode_viewport,
    diagnostic_capture_pixel_output, diagnostic_capture_vertex_position,
    diagnostic_compute_texture_dispatch_is_eligible, diagnostic_context_register_snapshot,
    diagnostic_disable_culling, diagnostic_disable_depth, diagnostic_disable_depth_and_culling,
    diagnostic_dump_complex_msaa, diagnostic_dump_compute_texture_minimum_dispatch,
    diagnostic_dump_depth_export_shader, diagnostic_dump_gbuffer, diagnostic_dump_glyph_draw,
    diagnostic_dump_shadow_consumer, diagnostic_dump_target_address,
    diagnostic_dump_target_address_range, diagnostic_dump_target_before_pixel_shader,
    diagnostic_dump_target_index_count, diagnostic_dump_target_match_index,
    diagnostic_dump_target_minimum_draw, diagnostic_dump_target_mrt_min,
    diagnostic_dump_target_pixel_instruction_count, diagnostic_dump_target_pixel_shader,
    diagnostic_dump_video_chain_consumer, diagnostic_dump_video_draw,
    diagnostic_dump_video_target_consumer, diagnostic_f32_bits_magnitude_nonzero,
    diagnostic_force_msaa_vertex_pattern, diagnostic_global_binding_head,
    diagnostic_note_full_resolution_mrt_draw, diagnostic_overlapping_global_writers,
    diagnostic_overlapping_memory_copies, diagnostic_overlapping_texture_writers,
    diagnostic_related_texture_writers, diagnostic_scanout_writer_slot,
    diagnostic_scanout_writer_target_selected, diagnostic_shader_summary,
    diagnostic_target_after_mrt_ready, diagnostic_target_draw_filter,
    diagnostic_texture_depth_plane_summaries, diagnostic_texture_magnitude_nonzero_texels,
    diagnostic_texture_numeric_range, diagnostic_texture_writer, diagnostic_texture_writer_enabled,
    diagnostic_trace_all_target_contents, diagnostic_trace_target_contents,
    diagnostic_trace_target_maximum_draw, diagnostic_trace_target_stride,
    diagnostic_uploaded_global_binding_head, diagnostic_use_draw_merged_wave_counts,
    diagnostic_video_chain_selected_draw, dump_msl_compute_filter_matches, dump_shader_pair_msl,
    evaluation_samples_glyph_atlas, evaluation_samples_large_linear_video,
    evaluation_samples_layered_comparison, fragmentless_pixel_evaluation, guest_frame_fingerprint,
    guest_texture_descriptors_overlap, guest_texture_descriptors_share_storage, half_float_to_f32,
    image_resource_descriptor_is_null, initial_depth_clear_for_target,
    initial_render_target_clear_color, is_storage_image_opcode, is_writable_image_opcode,
    metal_nonindexed_vertex_count, metal_primitive_type, metal_uncompressed_bytes_per_pixel,
    native_texture_capture_to_bgra, ngg_mesh_base_vertex, prepare_indexed_ngg_indices,
    prepare_indexed_primitive, prepare_nonindexed_ngg_indices, read_draw_indices,
    read_texture_array_linear, read_texture_linear, read_texture_volume_linear,
    rearm_color_metadata_clear, rearm_depth_metadata_clear, record_diagnostic_global_writers,
    record_diagnostic_scanout_writer, record_diagnostic_texture_writer, register_value,
    render_target_cmask_address, render_target_fast_clear_color,
    resident_color_view_sample_is_eligible, resident_depth_sample_is_eligible,
    resident_same_image_sample_is_eligible, sampled_image_bindings,
    shader_scalar_metadata_for_registers, should_record_diagnostic_target_writer,
    snapshot_guest_texture_raw, spans_overlap, storage_image_requires_mip_hierarchy,
    trace_glyph_atlas, trace_image_descriptors, trace_msaa_writes, trace_scanout_writes,
    log_wave64_hazard_census, trace_target_sampling, upload_image_textures,
    validate_compute_launch_for_metal, video_target_binding_summary, wave64_split_blockers,
};
use metal::{
    MTLClearColor, MTLCullMode, MTLIndexType, MTLLoadAction, MTLOrigin, MTLPixelFormat,
    MTLPrimitiveType, MTLRenderStages, MTLResourceOptions, MTLResourceUsage, MTLScissorRect,
    MTLSize, MTLTextureType, MTLViewport, RenderPassDescriptor,
};
use remu_gpu::IndexedDraw;
use std::collections::{HashMap, HashSet};
use std::ffi::c_void;
use std::path::Path;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, OnceLock};
use std::time::Instant;
mod compute;
mod indirect;

// metal-rs 0.33 predates the Mesh named constant, but its bitflags retain unknown Apple SDK bits.
const MTL_RENDER_STAGE_MESH_BITS: u64 = 1 << 4;

pub(super) fn export_resource_render_stage(uses_mesh_stage: bool) -> MTLRenderStages {
    if uses_mesh_stage {
        MTLRenderStages::from_bits_retain(MTL_RENDER_STAGE_MESH_BITS)
    } else {
        MTLRenderStages::Vertex
    }
}

pub(super) const fn ngg_mesh_dispatch_required(
    scissor: GuestScissor,
    instance_count: u32,
    has_meshlets: bool,
) -> bool {
    !matches!(scissor, GuestScissor::Empty) && instance_count != 0 && has_meshlets
}

pub(super) const fn metal_scissor_rect_for_draw(
    scissor: GuestScissor,
    render_width: u32,
    render_height: u32,
) -> Option<MTLScissorRect> {
    match scissor {
        GuestScissor::Full => Some(MTLScissorRect {
            x: 0,
            y: 0,
            width: render_width as u64,
            height: render_height as u64,
        }),
        GuestScissor::Rect(rect) => Some(rect),
        GuestScissor::Empty => None,
    }
}

impl MetalPresenter {
    #[allow(clippy::too_many_lines)]

    /// Builds the pixel stage's runtime heap texture tables (stage 3 of
    /// docs/design/2026-08-13-runtime-image-descriptor-fetch.md).
    ///
    /// Each table walks its SMEM binding's guest span in 32-byte steps from the binding's own
    /// base phase and stops at the first slot that does not decode as a texture: the measured
    /// resource blobs are mixed-width records, so a fixed-window walk would misparse past the
    /// table's end. Slots resolve through live residents first; unresolved slots bind a black
    /// placeholder and report once, so a miss reads as a named gap rather than garbage.
    fn build_pixel_heap_texture_tables(
        &mut self,
        device: &metal::Device,
        pixel_function: &metal::Function,
        evaluation: &remu_shader::ShaderEvaluation,
        tables: &[(u32, u32)],
    ) -> Result<Vec<(u32, metal::Buffer, Vec<metal::Texture>)>, String> {
        const HEAP_TABLE_SLOTS: u64 = 16;
        let mut built = Vec::with_capacity(tables.len());
        for (global_index, buffer_slot) in tables {
            let binding = evaluation
                .global_memory_bindings
                .get(*global_index as usize)
                .ok_or("heap texture table names a missing global binding")?;
            let base = binding.base_address;
            // Content-hash the table's slot bytes: identical tables recur at hundreds of
            // guest addresses (bump-allocated per-draw blobs), and rebuilding per draw feeds
            // the presenter-livelock class this path must never worsen.
            let mut content = [0_u8; 512];
            let content_known =
                remu_memory::read_guest_shared_backing(base, &mut content);
            if content_known {
                let mut hash = 0xcbf2_9ce4_8422_2325_u64;
                for byte in content {
                    hash ^= u64::from(byte);
                    hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
                }
                hash ^= u64::from(*buffer_slot);
                if let Some((buffer, textures)) = self.heap_table_cache.get(&hash) {
                    built.push((*buffer_slot, buffer.clone(), textures.clone()));
                    continue;
                }
                self.pending_heap_table_hash = Some(hash);
            } else {
                self.pending_heap_table_hash = None;
            }
            let argument_encoder =
                pixel_function.new_argument_encoder(u64::from(*buffer_slot));
            let argument_buffer = device.new_buffer(
                argument_encoder.encoded_length(),
                metal::MTLResourceOptions::StorageModeShared,
            );
            argument_encoder.set_argument_buffer(&argument_buffer, 0);
            let mut resolved: Vec<metal::Texture> = Vec::new();
            let mut walking = true;
            let placeholder = self.heap_table_placeholder_texture(device)?;
            for slot in 0..HEAP_TABLE_SLOTS {
                let mut texture: Option<metal::Texture> = None;
                if walking {
                    let mut bytes = [0_u8; 32];
                    if remu_memory::read_guest_shared_backing(base + slot * 32, &mut bytes) {
                        let words: [u32; 8] = std::array::from_fn(|index| {
                            u32::from_le_bytes(
                                bytes[index * 4..index * 4 + 4].try_into().expect("dword"),
                            )
                        });
                        match super::decode::decode_image_descriptor(&words) {
                            Ok(decoded) => {
                                texture = self
                                    .resident_guest_textures
                                    .get_mut(&decoded.guest)
                                    .and_then(|resident| {
                                        super::texture::resident_texture_for_sample(
                                            resident, decoded,
                                        )
                                    });
                                if texture.is_none() {
                                    // Compatible residents next (colour-view/same-image), then
                                    // the sampled cache, then a GPU-tiler upload from guest
                                    // RAM: the descriptor is a valid T# naming a real asset,
                                    // and a placeholder here multiplies the scene to black.
                                    let compatible = self
                                        .resident_guest_textures
                                        .values()
                                        .find(|resident| {
                                            super::texture::resident_color_view_sample_is_eligible(
                                                resident, decoded, false,
                                            ) || super::texture::resident_same_image_sample_is_eligible(
                                                resident, decoded, false,
                                            )
                                        })
                                        .map(|resident| resident.descriptor);
                                    if let Some(descriptor) = compatible
                                        && let Some(resident) =
                                            self.resident_guest_textures.get_mut(&descriptor)
                                    {
                                        texture =
                                            super::texture::resident_texture_for_sample(
                                                resident, decoded,
                                            );
                                    }
                                }
                                if texture.is_none() {
                                    let key = super::texture::sampled_texture_key(
                                        decoded, false,
                                    );
                                    texture = self
                                        .sampled_texture_cache
                                        .get(&key)
                                        .map(|cached| cached.texture.clone());
                                    if texture.is_none() {
                                        let queue = self.active_guest_command_queue()?.clone();
                                        let command_buffer = queue.new_command_buffer();
                                        match super::texture::create_sampled_guest_texture_from_guest_gpu(
                                            device,
                                            command_buffer,
                                            decoded,
                                            false,
                                            false,
                                        ) {
                                            Ok(created) => {
                                                command_buffer.commit();
                                                texture = Some(created);
                                            }
                                            Err(error) => {
                                                static LOGGED: std::sync::atomic::AtomicUsize =
                                                    std::sync::atomic::AtomicUsize::new(0);
                                                let sequence = LOGGED.fetch_add(
                                                    1,
                                                    std::sync::atomic::Ordering::Relaxed,
                                                ) + 1;
                                                if sequence <= 8
                                                    || sequence.is_power_of_two()
                                                {
                                                    eprintln!(
                                                        "[REMu][HEAP-TABLE] sequence={sequence} slot upload failed: {error}"
                                                    );
                                                }
                                            }
                                        }
                                    }
                                }
                                if texture.is_none() {
                                    static LOGGED: std::sync::atomic::AtomicUsize =
                                        std::sync::atomic::AtomicUsize::new(0);
                                    let sequence = LOGGED
                                        .fetch_add(1, std::sync::atomic::Ordering::Relaxed)
                                        + 1;
                                    if sequence <= 16 || sequence.is_power_of_two() {
                                        eprintln!(
                                            "[REMu][HEAP-TABLE] sequence={sequence} slot={slot} descriptor at {:#x} has no serving resident ({:?} {}x{})",
                                            base + slot * 32,
                                            decoded.guest.pixel_format,
                                            decoded.guest.width,
                                            decoded.guest.height,
                                        );
                                    }
                                }
                            }
                            Err(_) => walking = false,
                        }
                    } else {
                        walking = false;
                    }
                }
                let bound = texture.unwrap_or_else(|| placeholder.clone());
                argument_encoder.set_texture(slot, &bound);
                resolved.push(bound);
            }
            if let Some(hash) = self.pending_heap_table_hash.take() {
                self.heap_table_cache
                    .insert(hash, (argument_buffer.clone(), resolved.clone()));
                if self.heap_table_cache.len() > 2048 {
                    self.heap_table_cache.clear();
                }
            }
            built.push((*buffer_slot, argument_buffer, resolved));
        }
        Ok(built)
    }

    /// A 1x1 opaque-black texture for unresolved heap-table slots, created once per device.
    fn heap_table_placeholder_texture(
        &mut self,
        device: &metal::Device,
    ) -> Result<metal::Texture, String> {
        if let Some(texture) = self.heap_table_placeholder.as_ref() {
            return Ok(texture.clone());
        }
        let descriptor = metal::TextureDescriptor::new();
        descriptor.set_width(1);
        descriptor.set_height(1);
        descriptor.set_pixel_format(metal::MTLPixelFormat::RGBA8Unorm);
        descriptor.set_storage_mode(metal::MTLStorageMode::Shared);
        let texture = device.new_texture(&descriptor);
        texture.replace_region(
            metal::MTLRegion {
                origin: metal::MTLOrigin { x: 0, y: 0, z: 0 },
                size: metal::MTLSize {
                    width: 1,
                    height: 1,
                    depth: 1,
                },
            },
            0,
            [0_u8, 0, 0, 255].as_ptr().cast(),
            4,
        );
        self.heap_table_placeholder = Some(texture.clone());
        Ok(texture)
    }

    pub(super) fn execute_compiled_draw(
        &mut self,
        draw: &IndexedDraw,
        key: &GraphicsPipelineKey,
        shared_event_texture_overrides: &[SharedEventDrawTextureOverride],
    ) -> Result<(), String> {
        self.retire_completed_guest_batches();
        let started = Instant::now();
        let ack_phase_timing = self.shared_event_ack_phase_timing_enabled();
        let trace = draw.sequence <= 4 || draw.sequence.is_power_of_two();
        let targets = decode_render_targets(draw)?;
        let depth = decode_depth_target(draw)?;
        let raster = decode_raster_state(draw)?;
        let diagnostic_depth_draw = diagnostic_dump_depth_export_shader().is_some_and(|expected| {
            (expected == 0 || expected == draw.export_shader_address)
                && targets.is_empty()
                && draw.index_count == 6
                && depth.is_some_and(|depth| {
                    depth.write_enabled
                        && depth.texture.width == 1_920
                        && depth.texture.height == 1_080
                })
        });
        let diagnostic_gbuffer_program = diagnostic_dump_gbuffer()
            || diagnostic_dump_target_pixel_instruction_count().is_some_and(|count| {
                draw.pixel_shader
                    .as_ref()
                    .is_some_and(|shader| shader.program.instructions.len() == count)
            });
        let diagnostic_mrt_draw = diagnostic_dump_target_mrt_min().is_some_and(|minimum| {
            targets.len() >= minimum
                && targets
                    .iter()
                    .all(|target| target.texture.width >= 1_280 && target.texture.height >= 720)
        });
        let diagnostic_gbuffer_draw = diagnostic_mrt_draw
            || (diagnostic_gbuffer_program
                && draw.index_count == 6
                && !targets.is_empty()
                && targets
                    .iter()
                    .all(|target| target.texture.width == 1_920 && target.texture.height == 1_080)
                && (targets.len() >= 4 || draw.pixel_shader_address >= 0x8030_0000_00));
        if diagnostic_dump_gbuffer()
            && draw.pixel_shader_address >= 0x8030_0000_00
            && draw.index_count == 6
        {
            static EXECUTED_GBUFFER_CANDIDATES: AtomicUsize = AtomicUsize::new(0);
            let candidate = EXECUTED_GBUFFER_CANDIDATES.fetch_add(1, Ordering::Relaxed);
            if candidate < 16 {
                eprintln!(
                    "[REMu][METAL][GBUFFER-EXECUTE] candidate={candidate} draw={} es={:#x} ps={:#x} attachments={} selected={diagnostic_gbuffer_draw}",
                    draw.sequence,
                    draw.export_shader_address,
                    draw.pixel_shader_address,
                    targets.len(),
                );
            }
        }
        let color_control = register_value(&draw.context_registers, CB_COLOR_CONTROL)
            .unwrap_or(DEFAULT_CB_COLOR_CONTROL);
        let color_operation = decode_color_operation_mode(color_control)?;
        if color_operation_is_raster_noop(color_operation, depth.is_some()) {
            // MODE=Disable turns off color processing. With no active depth target there is
            // nothing for the draw to rasterize. In particular, AGC uses position-only rectangle
            // programs for such trigger/no-op packets while leaving an unrelated pixel program
            // bound; executing that program would overwrite the target.
            super::stats::NOOP_DRAWS_RASTER_DISABLED.fetch_add(1, Ordering::Relaxed);
            super::stats::note_noop_draw("gpu:noop-raster-disabled");
            return Ok(());
        }
        if matches!(
            color_operation,
            ColorOperationMode::Resolve
                | ColorOperationMode::EliminateFastClear
                | ColorOperationMode::FmaskDecompress
        ) {
            // CB_COLOR_CONTROL special operations use a draw packet only as a command-processor
            // trigger; its attached shader is not rasterized. Executing the trigger shader as a
            // normal draw overwrites the color target. Ordinary passes maintain a single-sample
            // MRT0 resident. Resolve transfers that resident to MRT1; EliminateFastClear
            // materializes the logical CMASK clear before later geometry loads the attachment.
            super::stats::NOOP_DRAWS_COLOR_OPERATION.fetch_add(1, Ordering::Relaxed);
            super::stats::note_noop_draw("gpu:noop-color-operation");
            if self.active_open_guest_batch_mut().is_some_and(|batch| {
                batch.end_compute_pass();
                batch.end_render_pass()
            }) {
                self.statistics.render_pass_ends_for_target_change = self
                    .statistics
                    .render_pass_ends_for_target_change
                    .saturating_add(1);
                self.statistics.render_pass_ends_for_color_operation = self
                    .statistics
                    .render_pass_ends_for_color_operation
                    .saturating_add(1);
            }
            let operation_targets = match color_operation {
                ColorOperationMode::Resolve => {
                    let (source, destination) = self.execute_color_resolve(draw)?;
                    Some((source, Some(destination)))
                }
                ColorOperationMode::EliminateFastClear => self
                    .execute_eliminate_fast_clear(draw)?
                    .map(|target| (target, None)),
                _ => None,
            };
            if trace_target_sampling() {
                static LOGGED_COLOR_OPERATIONS: AtomicUsize = AtomicUsize::new(0);
                let sequence = LOGGED_COLOR_OPERATIONS.fetch_add(1, Ordering::Relaxed) + 1;
                let target_minimum = diagnostic_dump_target_minimum_draw();
                let near_target_minimum = draw.sequence >= target_minimum
                    && draw.sequence < target_minimum.saturating_add(64);
                if sequence <= 16 || sequence.is_power_of_two() || near_target_minimum {
                    eprintln!(
                        "[REMu][METAL][COLOR-OP] draw={} operation={color_operation:?} control={color_control:#010x} targets={targets:?} operated={operation_targets:?}",
                        draw.sequence,
                    );
                }
            }
            return Ok(());
        }
        self.capture_scanout_before_pixel_shader(draw, &targets)?;
        if trace_scanout_writes() && draw.sequence >= diagnostic_dump_target_minimum_draw() {
            static LOGGED_SCANOUT_WRITES: AtomicUsize = AtomicUsize::new(0);
            for target in &targets {
                if self
                    .scanout_snapshot_textures
                    .contains_key(&target.texture.address)
                {
                    let write = LOGGED_SCANOUT_WRITES.fetch_add(1, Ordering::Relaxed);
                    if write < 512 {
                        eprintln!(
                            "[REMu][METAL][SCANOUT-WRITE] index={write} draw={} es={:#x} ps={:#x} ps_insns={} indexed={} count={} instances={} primitive={:#x} target={:?} depth={depth:?} pipeline_targets={:?} images={}/{} globals={}/{}",
                            draw.sequence,
                            draw.export_shader_address,
                            draw.pixel_shader_address,
                            draw.pixel_shader
                                .as_ref()
                                .map_or(0, |shader| shader.program.instructions.len()),
                            draw.indexed,
                            draw.index_count,
                            draw.instance_count,
                            draw.primitive_type,
                            target,
                            key.targets,
                            draw.export_shader_evaluation
                                .as_ref()
                                .map_or(0, |evaluation| evaluation.image_bindings.len()),
                            draw.pixel_shader_evaluation
                                .as_ref()
                                .map_or(0, |evaluation| evaluation.image_bindings.len()),
                            draw.export_shader_evaluation
                                .as_ref()
                                .map_or(0, |evaluation| evaluation.global_memory_bindings.len()),
                            draw.pixel_shader_evaluation
                                .as_ref()
                                .map_or(0, |evaluation| evaluation.global_memory_bindings.len()),
                        );
                    }
                }
            }
        }
        let diagnostic_complex_msaa_raster = targets.iter().any(|target| target.sample_count > 1)
            && draw
                .pixel_shader
                .as_ref()
                .is_some_and(|shader| shader.program.instructions.len() == 370);
        // The combined full-raster diagnostic is intended to probe the complex gameplay pass.
        // Scoping it here avoids turning every tiny MSAA clear/copy draw into catastrophic
        // full-screen overdraw during loading. The independent cull/depth probes retain their
        // original all-draw behavior.
        let diagnostic_disable_combined_raster = diagnostic_disable_depth_and_culling();
        let diagnostic_disable_culling_for_draw = diagnostic_disable_culling()
            && (!diagnostic_disable_combined_raster || diagnostic_complex_msaa_raster);
        let diagnostic_disable_depth_for_draw = diagnostic_disable_depth()
            && (!diagnostic_disable_combined_raster || diagnostic_complex_msaa_raster);
        if trace_target_sampling() {
            let target_minimum = diagnostic_dump_target_minimum_draw();
            let near_target_minimum = draw.sequence >= target_minimum
                && draw.sequence < target_minimum.saturating_add(128);
            if near_target_minimum
                && (depth.is_some()
                    || targets.iter().any(|target| {
                        target.texture.width == 1_920 && target.texture.height == 1_080
                    }))
            {
                eprintln!(
                    "[REMu][METAL][LATE-DRAW] draw={} indexed={} count={} instances={} primitive={:#x} es={:#x}/{} ps={:#x}/{} color_control={color_control:#010x} operation={color_operation:?} raster={raster:?} targets={targets:?} depth={depth:?}",
                    draw.sequence,
                    draw.indexed,
                    draw.index_count,
                    draw.instance_count,
                    draw.primitive_type,
                    draw.export_shader_address,
                    draw.export_shader
                        .as_ref()
                        .map_or(0, |shader| shader.program.instructions.len()),
                    draw.pixel_shader_address,
                    draw.pixel_shader
                        .as_ref()
                        .map_or(0, |shader| shader.program.instructions.len()),
                );
            }
            // K1: lock-free bounded sampler (was a mutex-guarded HashSet dedup).
            static LOGGED_TARGETS: AtomicUsize = AtomicUsize::new(0);
            for target in &targets {
                let descriptor = target.texture;
                if descriptor.width == 1_920
                    && descriptor.height == 1_080
                    && (LOGGED_TARGETS.fetch_add(1, Ordering::Relaxed) < 256 || near_target_minimum)
                {
                    let primary = target.guest_slot * CB_COLOR_REGISTER_STRIDE;
                    let fmask = register_value(&draw.context_registers, CB_COLOR0_FMASK + primary);
                    let fmask_ext = register_value(
                        &draw.context_registers,
                        CB_COLOR0_FMASK_BASE_EXT + target.guest_slot,
                    );
                    let fmask_address = fmask
                        .zip(fmask_ext)
                        .map(|(base, ext)| (u64::from(ext & 0xff) << 40) | (u64::from(base) << 8));
                    eprintln!(
                        "[REMu][METAL][TARGET-WRITE] draw={} es={:#x} ps={:#x} ps_insns={} slot={} address={:#x} fmask={fmask_address:x?} extent={}x{} pitch={} format={:?} tile={} mask={:#x} samples={}",
                        draw.sequence,
                        draw.export_shader_address,
                        draw.pixel_shader_address,
                        draw.pixel_shader
                            .as_ref()
                            .map_or(0, |shader| shader.program.instructions.len()),
                        target.guest_slot,
                        descriptor.address,
                        descriptor.width,
                        descriptor.height,
                        descriptor.pitch,
                        descriptor.pixel_format,
                        descriptor.tile_mode,
                        target.write_mask,
                        target.sample_count,
                    );
                }
            }
            static GAMEPLAY_TARGET: OnceLock<Mutex<Option<GuestTextureDescriptor>>> =
                OnceLock::new();
            let gameplay_target = {
                let mut gameplay_target = GAMEPLAY_TARGET
                    .get_or_init(|| Mutex::new(None))
                    .lock()
                    .unwrap_or_else(std::sync::PoisonError::into_inner);
                if gameplay_target.is_none()
                    && draw
                        .pixel_shader
                        .as_ref()
                        .is_some_and(|shader| shader.program.instructions.len() == 370)
                {
                    *gameplay_target = targets
                        .iter()
                        .find(|target| target.sample_count > 1)
                        .map(|target| target.texture);
                }
                *gameplay_target
            };
            if let Some(gameplay_target) = gameplay_target {
                static WRITES: AtomicUsize = AtomicUsize::new(0);
                for target in &targets {
                    if target.texture.address == gameplay_target.address {
                        let write = WRITES.fetch_add(1, Ordering::Relaxed);
                        if write < 128 {
                            eprintln!(
                                "[REMu][METAL][GAMEPLAY-WRITE] draw={} es={:#x} ps={:#x} ps_insns={} samples={} format={:?} mask={:#x}",
                                draw.sequence,
                                draw.export_shader_address,
                                draw.pixel_shader_address,
                                draw.pixel_shader
                                    .as_ref()
                                    .map_or(0, |shader| shader.program.instructions.len()),
                                target.sample_count,
                                target.texture.pixel_format,
                                target.write_mask,
                            );
                        }
                    }
                }
            }
        }
        if !self.compiled_shader_pairs.contains_key(key) {
            return Err("shader pair has no compiled Metal pipeline".to_owned());
        }
        let export_evaluation = draw
            .export_shader_evaluation
            .as_ref()
            .ok_or("draw has no export resource evaluation")?;
        let pixel_evaluation = match (
            draw.pixel_shader.as_ref(),
            draw.pixel_shader_evaluation.as_ref(),
        ) {
            (Some(_), Some(evaluation)) => evaluation.as_ref(),
            (None, None) if targets.is_empty() && depth.is_some() => {
                fragmentless_pixel_evaluation()
            }
            (None, None) => {
                return Err(
                    "fragmentless draw requires an active depth target and no color targets"
                        .to_owned(),
                );
            }
            (Some(_), None) => return Err("draw has no pixel resource evaluation".to_owned()),
            (None, Some(_)) => {
                return Err("draw has pixel resources but no decoded pixel shader".to_owned());
            }
        };
        if trace_msaa_writes() && draw.sequence >= diagnostic_dump_target_minimum_draw() {
            static LOGGED_MULTISAMPLE_WRITES: AtomicUsize = AtomicUsize::new(0);
            for target in targets.iter().filter(|target| target.sample_count > 1) {
                let write = LOGGED_MULTISAMPLE_WRITES.fetch_add(1, Ordering::Relaxed);
                if write < 2_048 {
                    let blend = key
                        .targets
                        .iter()
                        .find(|state| state.guest_slot == target.guest_slot)
                        .map(|state| state.blend);
                    let color_control = register_value(&draw.context_registers, CB_COLOR_CONTROL)
                        .unwrap_or_default();
                    eprintln!(
                        "[REMu][METAL][MSAA-WRITE] draw={} es={:#x} ps={:#x} instructions={} indexed={} count={} instances={} target={:#x} samples={} depth={} sampled={} globals={} color_control={color_control:#010x} blend={blend:?}",
                        draw.sequence,
                        draw.export_shader_address,
                        draw.pixel_shader_address,
                        draw.pixel_shader
                            .as_ref()
                            .map_or(0, |shader| shader.program.instructions.len()),
                        draw.indexed,
                        draw.index_count,
                        draw.instance_count,
                        target.texture.address,
                        target.sample_count,
                        depth.is_some(),
                        pixel_evaluation.image_bindings.len(),
                        pixel_evaluation.global_memory_bindings.len(),
                    );
                }
            }
        }
        if diagnostic_texture_writer_enabled() {
            let diagnostic_sampled = sampled_image_bindings(pixel_evaluation)
                .filter_map(|binding| decode_image_descriptor(&binding.resource_descriptor).ok())
                .map(|descriptor| descriptor.guest)
                .collect::<Vec<_>>();
            for target in &targets {
                record_diagnostic_texture_writer(
                    target.texture,
                    "draw",
                    draw.sequence,
                    draw.pixel_shader_address,
                    diagnostic_sampled.clone(),
                );
            }
        }
        record_diagnostic_global_writers(
            "vertex",
            draw.sequence,
            draw.export_shader_address,
            export_evaluation,
        );
        record_diagnostic_global_writers(
            "pixel",
            draw.sequence,
            draw.pixel_shader_address,
            pixel_evaluation,
        );
        let device = self
            .device
            .as_ref()
            .ok_or("Metal device is unavailable")?
            .clone();
        let queue = self.active_guest_command_queue()?;
        // Color attachments determine the framebuffer extent when present. Unreal's early-Z and
        // masked-depth passes intentionally program CB_TARGET_MASK=0, so a depth-only pass derives
        // its extent from DB instead. Rejecting those passes leaves the later depth-equal base
        // pass with an empty depth buffer and makes otherwise valid 3D scenes entirely black.
        let render_width = targets
            .iter()
            .map(|target| (target.texture.width >> target.mip_level).max(1))
            .min()
            .or_else(|| depth.map(|depth| depth.texture.width))
            .ok_or("draw has neither a color nor depth attachment")?;
        let render_height = targets
            .iter()
            .map(|target| (target.texture.height >> target.mip_level).max(1))
            .min()
            .or_else(|| depth.map(|depth| depth.texture.height))
            .ok_or("draw has neither a color nor depth attachment")?;
        let mut target_textures = Vec::with_capacity(targets.len());
        let mut target_resolve_textures = Vec::with_capacity(targets.len());
        let mut initial_target_clear_colors = Vec::with_capacity(targets.len());
        // Residency may serve a target through a pitch-mismatched alias of the decoded
        // descriptor. Keep the key it chose so the post-draw write reaches that resident.
        let mut target_resident_descriptors = Vec::with_capacity(targets.len());
        for target in &targets {
            let resolved_texture = if target.direct_scanout {
                let texture = self.ensure_direct_scanout_texture(&device, target.texture);
                initial_target_clear_colors.push(
                    self.direct_scanout_needs_initial_clear(target.texture.address)
                        .then_some(MTLClearColor::new(0.0, 0.0, 0.0, 1.0)),
                );
                target_resident_descriptors.push(None);
                texture
            } else {
                let has_resident_storage = self.resident_guest_textures.values().any(|resident| {
                    resident.serves_reads()
                        && guest_texture_descriptors_share_storage(
                            resident.descriptor,
                            target.texture,
                        )
                });
                let metadata_clear_pending = consume_color_metadata_clear(
                    &mut self.color_metadata_clears,
                    render_target_cmask_address(target, &draw.context_registers)?,
                    target.base_layer,
                );
                let initial_clear_color = initial_render_target_clear_color(
                    target,
                    &draw.context_registers,
                    has_resident_storage,
                    metadata_clear_pending,
                );
                initial_target_clear_colors.push(initial_clear_color);
                if !has_resident_storage && (target.dcc_enabled || metadata_clear_pending) {
                    static DISCARDED_DCC_TARGETS: AtomicUsize = AtomicUsize::new(0);
                    let sequence = DISCARDED_DCC_TARGETS.fetch_add(1, Ordering::Relaxed) + 1;
                    if sequence <= 16 || sequence.is_power_of_two() {
                        eprintln!(
                            "[REMu][METAL] materializing compressed color target {} at {:#x} as uncompressed Metal storage dcc={} fast_clear={} initial_clear={} sequence={sequence}",
                            target.guest_slot,
                            target.texture.address,
                            target.dcc_enabled,
                            target.fast_clear_enabled,
                            initial_clear_color.is_some(),
                        );
                    }
                }
                let selected = self
                    .ensure_layered_render_target_texture(
                        &device,
                        *target,
                        initial_clear_color.is_some(),
                    )
                    .map_err(|error| {
                        format!(
                            "render target {} guest memory is unreadable: {error}",
                            target.guest_slot
                        )
                    })?;
                target_resident_descriptors.push(Some(selected.descriptor));
                selected.texture
            };
            if target.sample_count > 1 {
                let (multisample, clear_multisample) = self.ensure_multisample_texture(
                    &device,
                    target.texture,
                    target.sample_count,
                    target.texture.width,
                    target.texture.height,
                )?;
                target_textures.push(multisample);
                target_resolve_textures.push(Some(resolved_texture));
                if clear_multisample {
                    let clear_color =
                        render_target_fast_clear_color(target, &draw.context_registers)
                            .unwrap_or_else(|| MTLClearColor::new(0.0, 0.0, 0.0, 1.0));
                    *initial_target_clear_colors
                        .last_mut()
                        .expect("target clear state was just added") = Some(clear_color);
                } else {
                    *initial_target_clear_colors
                        .last_mut()
                        .expect("target clear state was just added") = None;
                }
            } else {
                target_textures.push(resolved_texture);
                target_resolve_textures.push(None);
            }
        }
        let depth_had_resident_storage = depth.is_some_and(|depth| {
            self.resident_guest_textures.values().any(|resident| {
                resident.serves_reads()
                    && guest_texture_descriptors_share_storage(resident.descriptor, depth.texture)
            })
        });
        let resolved_depth = depth
            .map(|depth| {
                self.ensure_depth_render_target_texture(&device, depth, render_width, render_height)
                    .map_err(|error| format!("depth target guest memory is unreadable: {error}"))
            })
            .transpose()?;
        let depth_resident_descriptor = resolved_depth.as_ref().map(|resolved| resolved.descriptor);
        let resolved_depth_texture = resolved_depth.map(|resolved| resolved.texture);
        let (depth_texture, depth_resolve_texture, clear_uninitialized_depth) =
            if let (Some(depth), Some(resolved)) = (depth, resolved_depth_texture) {
                if depth.sample_count > 1 {
                    let (multisample, clear) = self.ensure_multisample_texture(
                        &device,
                        depth.texture,
                        depth.sample_count,
                        render_width,
                        render_height,
                    )?;
                    (Some(multisample), Some(resolved), clear)
                } else {
                    // A fresh Gen5 Z allocation is generally represented by HTILE metadata and
                    // its base tiled pixels are not meaningful until decompressed. Importing
                    // those bytes as ordinary depth made a first read-only pass reject arbitrary
                    // geometry. Materialize first use with the programmed DB clear, just as the
                    // hardware expands an HTILE fast clear.
                    (Some(resolved), None, !depth_had_resident_storage)
                }
            } else {
                (None, None, false)
            };

        // A sampled SRD can name a demoted render-graph tenant even when it is not a storage
        // opcode. Restore that exact tenant on the GPU before cache selection gets a chance to
        // decode stale RAM. Malformed SRDs retain the ordinary fallback-image path below.
        for evaluation in [export_evaluation, pixel_evaluation] {
            for binding in evaluation
                .image_bindings
                .iter()
                .filter(|binding| !image_resource_descriptor_is_null(&binding.resource_descriptor))
            {
                if let Ok(descriptor) = decode_image_descriptor(&binding.resource_descriptor)
                    && !shared_event_texture_overrides
                        .iter()
                        .any(|retained| retained.matches(descriptor))
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
        }

        for evaluation in [export_evaluation, pixel_evaluation] {
            for binding in evaluation
                .image_bindings
                .iter()
                .filter(|binding| is_storage_image_opcode(binding.opcode))
                .filter(|binding| !image_resource_descriptor_is_null(&binding.resource_descriptor))
            {
                let writable = is_writable_image_opcode(binding.opcode);
                let descriptor = match decode_image_descriptor(&binding.resource_descriptor) {
                    Ok(descriptor) => descriptor,
                    // Invalid or unbound read-only images are bound to the zero fallback by
                    // `upload_image_textures` below. They have no guest allocation to restore,
                    // alias-check, or materialize during this residency prepass.
                    Err(_) if !writable => continue,
                    Err(error) => {
                        return Err(format!("image at PC {:#x}: {error}", binding.pc));
                    }
                };
                // ImageLoad is read-only. Preserve a GPU-current render/depth image when Metal
                // can expose the requested component or native Z view directly; materializing an
                // exact SRD first would synchronously write the attachment back to guest memory
                // and recreate it. Writable storage operations still require exact storage.
                let has_read_compatible_resident = !writable
                    && self.resident_guest_textures.values().any(|resident| {
                        resident_color_view_sample_is_eligible(resident, descriptor, false)
                            || resident_same_image_sample_is_eligible(resident, descriptor, false)
                            || resident_depth_sample_is_eligible(resident, descriptor)
                    });
                if !has_read_compatible_resident {
                    for target in &targets {
                        if descriptor.guest != target.texture
                            && spans_overlap(descriptor.span, target.span)
                        {
                            return Err(format!(
                                "{} image at PC {:#x} aliases render target {} through an incompatible view",
                                if writable {
                                    "writable"
                                } else {
                                    "read-only storage"
                                },
                                binding.pc,
                                target.guest_slot
                            ));
                        }
                    }
                    if let Some(depth) = depth
                        && descriptor.guest != depth.texture
                        && spans_overlap(descriptor.span, depth.span)
                    {
                        return Err(format!(
                            "{} image at PC {:#x} aliases the depth target through an incompatible view",
                            if writable {
                                "writable"
                            } else {
                                "read-only storage"
                            },
                            binding.pc
                        ));
                    }
                }
                if descriptor.texture_type == 9 {
                    if storage_image_requires_mip_hierarchy(binding, descriptor) {
                        self.ensure_mipmapped_storage_texture(&device, descriptor)?;
                    } else if writable || !has_read_compatible_resident {
                        self.ensure_resident_texture_with_initial_contents(
                            &device,
                            descriptor.guest,
                            descriptor.span,
                            true,
                        )?;
                    }
                }
            }
        }
        self.synchronize_sampled_aliases(
            &[export_evaluation, pixel_evaluation],
            shared_event_texture_overrides,
        )?;

        self.ensure_active_open_guest_batch()?;
        self.prepare_depth_array_textures(&device, &[export_evaluation, pixel_evaluation])?;
        let export_buffers = self.upload_global_buffers(&device, export_evaluation)?;
        let pixel_buffers = self.upload_global_buffers(&device, pixel_evaluation)?;
        let activity_stamp = ResidentTextureActivityStamp {
            work: draw.sequence,
            flip: self.next_flip_sequence,
        };
        // GFX10 titles assemble per-draw descriptor tables with GPU work in the same frame; a
        // T# snapshot taken at decode time then reads as the slot's previous contents and the
        // sample would be refused. When the slot lies in a pending batch's writable output,
        // settle those batches first — writable-global writeback publishes the descriptor and
        // the upload path's late-resolve re-reads it — mirroring the hardware ordering where
        // the sample's memory dependency serializes through the producing dispatch.
        for evaluation in [export_evaluation, pixel_evaluation] {
            for binding in &evaluation.image_bindings {
                if binding.resource_descriptor.iter().all(|word| *word == 0)
                    || super::decode::decode_image_descriptor(&binding.resource_descriptor)
                        .is_ok()
                {
                    continue;
                }
                let Some(source) =
                    super::texture::descriptor_source_address(evaluation, &binding.resource_descriptor)
                else {
                    continue;
                };
                if self.any_guest_batch(|batch| {
                    super::submission::guest_batch_writes_overlap(batch, source, 32)
                }) {
                    static LOGGED: std::sync::atomic::AtomicUsize =
                        std::sync::atomic::AtomicUsize::new(0);
                    let sequence = LOGGED.fetch_add(1, std::sync::atomic::Ordering::Relaxed) + 1;
                    if sequence <= 16 || sequence.is_power_of_two() {
                        eprintln!(
                            "[REMu][METAL][DESCRIPTOR-SETTLE] sequence={sequence} draw={} pc={:#x} source={source:#x}",
                            draw.sequence, binding.pc,
                        );
                    }
                    self.wait_for_guest_command_batches()?;
                    break;
                }
            }
        }
        let export_textures = upload_image_textures(
            "vertex",
            &device,
            &queue,
            export_evaluation,
            &mut self.resident_guest_textures,
            &self.depth_array_textures,
            &self.multisample_textures,
            &mut self.sampled_texture_cache,
            &mut self.sampled_texture_write_watches,
            &mut self.fallback_texture_cache,
            &mut self.next_sampled_texture_use,
            activity_stamp,
            shared_event_texture_overrides,
        )?;
        let pixel_textures = upload_image_textures(
            "pixel",
            &device,
            &queue,
            pixel_evaluation,
            &mut self.resident_guest_textures,
            &self.depth_array_textures,
            &self.multisample_textures,
            &mut self.sampled_texture_cache,
            &mut self.sampled_texture_write_watches,
            &mut self.fallback_texture_cache,
            &mut self.next_sampled_texture_use,
            activity_stamp,
            shared_event_texture_overrides,
        )?;
        if remu_memory::debug_log_enabled() {
            let mut listed = String::new();
            for target in &targets {
                use std::fmt::Write as _;
                let _ = write!(
                    listed,
                    " tgt={:#x}/{:?}/{}x{}",
                    target.texture.address,
                    target.texture.pixel_format,
                    target.texture.width,
                    target.texture.height
                );
            }
            remu_memory::debug_log!(
                "draw",
                "seq={} es={:#x} ps={:#x} flip={}{}",
                draw.sequence,
                draw.export_shader_address,
                draw.pixel_shader_address,
                self.next_flip_sequence,
                listed
            );
        }
        let diagnostic_scanout_target = targets.iter().copied().find(|target| {
            diagnostic_scanout_writer_target_selected(
                draw.pixel_shader_address,
                target.texture.address,
                self.is_diagnostic_scanout_target(target),
                target.texture.width,
                target.texture.height,
            )
        });
        let diagnostic_scanout_writer_slot =
            diagnostic_scanout_target.and_then(|_| diagnostic_scanout_writer_slot(draw.sequence));
        if let (Some(slot), Some(target)) =
            (diagnostic_scanout_writer_slot, diagnostic_scanout_target)
        {
            eprintln!(
                "[REMu][METAL][SCANOUT-SOURCE] slot={slot} kind=target draw={} es={:#x} ps={:#x} target={target:?}",
                draw.sequence, draw.export_shader_address, draw.pixel_shader_address,
            );
            for (stage, evaluation, textures) in [
                (
                    "vertex",
                    export_evaluation.as_ref(),
                    export_textures.as_slice(),
                ),
                ("pixel", pixel_evaluation, pixel_textures.as_slice()),
            ] {
                for (binding_index, (binding, uploaded)) in canonical_texture_bindings(evaluation)
                    .into_iter()
                    .zip(textures)
                    .enumerate()
                {
                    let resident = uploaded
                        .selected_resident
                        .and_then(|descriptor| self.resident_guest_textures.get(&descriptor));
                    let resident_summary = resident.map(|resident| {
                        (
                            resident.descriptor,
                            resident.state,
                            resident.serves_reads(),
                            resident.gpu_written(),
                            resident.content_version,
                            resident.owned.byte_count(),
                            resident.gpu_authored.byte_count(),
                            resident.last_use,
                        )
                    });
                    eprintln!(
                        "[REMu][METAL][SCANOUT-SOURCE] slot={slot} kind=binding draw={} es={:#x} ps={:#x} target={target:?} stage={stage} binding={binding_index} pc={:#x} op={} words={:08x?} requested={:?} selected={resident_summary:?} native={:?}/{}x{}/mips{}/layers{}",
                        draw.sequence,
                        draw.export_shader_address,
                        draw.pixel_shader_address,
                        binding.pc,
                        binding.opcode,
                        binding.resource_descriptor,
                        uploaded.descriptor,
                        uploaded.texture.pixel_format(),
                        uploaded.texture.width(),
                        uploaded.texture.height(),
                        uploaded.texture.mipmap_level_count(),
                        uploaded.texture.array_length(),
                    );
                }
            }
        }
        let (
            pipeline,
            vertex_function,
            pixel_function,
            vertex_global_argument_buffer_index,
            pixel_global_argument_buffer_index,
            vertex_scalar_buffer_index,
            pixel_scalar_buffer_index,
            vertex_sampler_buffer_index,
            pixel_sampler_buffer_index,
            vertex_heap_texture_tables,
            pixel_heap_texture_tables,
            uses_mesh_stage,
            uses_procedural_rectangle_stage,
            vertex_msl_bytes,
            pixel_msl_bytes,
            vertex_dispatcher,
            pixel_dispatcher,
        ) = {
            let compiled = self
                .compiled_shader_pairs
                .get(key)
                .ok_or("shader pair has no compiled Metal pipeline")?;
            if let Some(directory) = std::env::var_os("REMU_DUMP_SCANOUT_MSL_DIR")
                && (targets.iter().any(|target| {
                    self.scanout_snapshot_textures
                        .contains_key(&target.texture.address)
                }) || diagnostic_dump_target_pixel_shader() == Some(draw.pixel_shader_address)
                    || diagnostic_depth_draw
                    || diagnostic_gbuffer_draw
                    || diagnostic_dump_target_pixel_instruction_count().is_some_and(|count| {
                        draw.pixel_shader
                            .as_ref()
                            .is_some_and(|shader| shader.program.instructions.len() == count)
                    }))
            {
                // K1: lock-free bounded sampler (was a mutex-guarded HashSet dedup).
                static DUMPED_SCANOUT_PAIRS: AtomicUsize = AtomicUsize::new(0);
                // General scanout tracing can exhaust its cap during the long title startup.
                // Preserve one independent slot for the first semantically selected G-buffer
                // draw so a relocated shader address cannot make the requested dump disappear.
                static DUMPED_DIAGNOSTIC_GBUFFER_PAIR: AtomicBool = AtomicBool::new(false);
                let general_slot = DUMPED_SCANOUT_PAIRS.fetch_add(1, Ordering::Relaxed) < 128;
                let diagnostic_gbuffer_slot = diagnostic_gbuffer_draw
                    && !DUMPED_DIAGNOSTIC_GBUFFER_PAIR.swap(true, Ordering::Relaxed);
                if general_slot || diagnostic_gbuffer_slot {
                    let sampled = sampled_image_bindings(pixel_evaluation)
                        .map(|binding| {
                            let descriptor =
                                decode_image_descriptor(&binding.resource_descriptor).ok();
                            (
                                binding.pc,
                                binding.opcode,
                                descriptor.map(|descriptor| descriptor.guest),
                                binding.sampler_descriptor.clone(),
                            )
                        })
                        .collect::<Vec<_>>();
                    eprintln!(
                        "[REMu][METAL][SCANOUT] draw={} es={:#x} ps={:#x} targets={targets:?} sampled={sampled:?}",
                        draw.sequence, draw.export_shader_address, draw.pixel_shader_address,
                    );
                    if let Err(error) = dump_shader_pair_msl(
                        Path::new(&directory),
                        draw.export_shader_address,
                        draw.pixel_shader_address,
                        &compiled.vertex.source,
                        &compiled.pixel.source,
                    ) {
                        eprintln!(
                            "[REMu][METAL] could not dump scanout MSL for es={:#x} ps={:#x}: {error}",
                            draw.export_shader_address, draw.pixel_shader_address,
                        );
                    }
                }
            }
            if diagnostic_dump_video_chain_consumer()
                && diagnostic_video_chain_selected_draw().load(Ordering::Relaxed) == draw.sequence
                && let Some(directory) = std::env::var_os("REMU_DUMP_VIDEO_CHAIN_MSL_DIR")
            {
                static DUMPED_VIDEO_CHAIN_PAIR: AtomicBool = AtomicBool::new(false);
                if !DUMPED_VIDEO_CHAIN_PAIR.swap(true, Ordering::Relaxed) {
                    if let Err(error) = dump_shader_pair_msl(
                        Path::new(&directory),
                        draw.export_shader_address,
                        draw.pixel_shader_address,
                        &compiled.vertex.source,
                        &compiled.pixel.source,
                    ) {
                        eprintln!(
                            "[REMu][METAL] could not dump video-chain MSL for es={:#x} ps={:#x}: {error}",
                            draw.export_shader_address, draw.pixel_shader_address,
                        );
                    } else {
                        eprintln!(
                            "[REMu][METAL][VIDEO-CHAIN-MSL] draw={} es={:#x} ps={:#x} vertex_bytes={} pixel_bytes={} vertex_capture={} pixel_capture={} directory={}",
                            draw.sequence,
                            draw.export_shader_address,
                            draw.pixel_shader_address,
                            compiled.vertex.source.len(),
                            compiled.pixel.source.len(),
                            compiled.vertex.source.contains("remu_debug_position"),
                            compiled.pixel.source.contains("remu_debug_pixel"),
                            Path::new(&directory).display(),
                        );
                    }
                }
            }
            (
                compiled.pipeline.clone(),
                compiled.vertex_function.clone(),
                compiled.pixel_function.clone(),
                compiled.vertex.bindings.global_argument_buffer_index,
                compiled.pixel.bindings.global_argument_buffer_index,
                compiled.vertex.bindings.initial_scalar_buffer_index,
                compiled.pixel.bindings.initial_scalar_buffer_index,
                compiled.vertex.bindings.sampler_argument_buffer_index,
                compiled.pixel.bindings.sampler_argument_buffer_index,
                compiled.vertex.bindings.heap_texture_tables.clone(),
                compiled.pixel.bindings.heap_texture_tables.clone(),
                compiled.uses_mesh_stage,
                compiled.uses_procedural_rectangle_stage,
                compiled.vertex.source.len(),
                compiled.pixel.source.len(),
                compiled.vertex.source.contains("while (active)"),
                compiled.pixel.source.contains("while (active)"),
            )
        };
        let indirect_arguments = if let Some(address) = draw.indirect_arguments_address {
            if draw.indexed {
                if !matches!(draw.index_size, 0 | 1) {
                    return Err(format!(
                        "draw {} uses GPU indirect arguments {address:#x} with unsupported \
                         {}-encoded index elements",
                        draw.sequence, draw.index_size,
                    ));
                }
                if register_value(&draw.context_registers, 0x2a5)
                    .is_some_and(|value| value & 1 != 0)
                {
                    return Err(format!(
                        "draw {} uses GPU indirect arguments {address:#x} with primitive restart; \
                         refusing the stale packet-time count because restart expansion requires \
                         a GPU index conversion",
                        draw.sequence,
                    ));
                }
            }
            let byte_count = if draw.indexed {
                5 * std::mem::size_of::<u32>()
            } else {
                4 * std::mem::size_of::<u32>()
            };
            let arguments = self.upload_ordered_indirect_arguments(&device, address, byte_count)?;
            let source_offset = arguments.offset;
            let rectangle_conversion =
                !uses_mesh_stage && !draw.indexed && matches!(draw.primitive_type, 7 | 0x11);
            let arguments = if rectangle_conversion {
                self.convert_rectangle_indirect_arguments(&device, &arguments)?
            } else {
                arguments
            };
            static LOGGED_INDIRECT_DRAWS: AtomicUsize = AtomicUsize::new(0);
            let sequence = LOGGED_INDIRECT_DRAWS.fetch_add(1, Ordering::Relaxed) + 1;
            if sequence <= 16 || sequence.is_power_of_two() {
                eprintln!(
                    "[REMu][METAL][INDIRECT-DRAW] sequence={sequence} draw={} indexed={} arguments={address:#x}+{byte_count:#x} observed=[count={},instances={},first={},base_vertex={}] source_offset={source_offset:#x} native_offset={:#x} rectangle_conversion={rectangle_conversion}",
                    draw.sequence,
                    draw.indexed,
                    draw.index_count,
                    draw.instance_count,
                    draw.index_offset,
                    draw.base_vertex,
                    arguments.offset,
                );
            }
            Some(arguments)
        } else {
            None
        };
        let indirect_index_buffer = if draw.indexed && indirect_arguments.is_some() {
            let (element_size, index_type) = match draw.index_size {
                0 => (std::mem::size_of::<u16>(), MTLIndexType::UInt16),
                1 => (std::mem::size_of::<u32>(), MTLIndexType::UInt32),
                _ => unreachable!("unsupported indirect index size was rejected above"),
            };
            let byte_count = usize::try_from(draw.index_buffer_count)
                .map_err(|_| "indirect index count does not fit usize")?
                .checked_mul(element_size)
                .ok_or("indirect index buffer byte count overflow")?;
            if draw.index_buffer_address < 0x10_000 || byte_count == 0 {
                return Err(format!(
                    "draw {} has invalid indirect index buffer {:#x}/{}",
                    draw.sequence, draw.index_buffer_address, draw.index_buffer_count,
                ));
            }
            Some((
                self.upload_ordered_indirect_arguments(
                    &device,
                    draw.index_buffer_address,
                    byte_count,
                )?,
                index_type,
            ))
        } else {
            None
        };
        let indirect_ngg = if uses_mesh_stage {
            indirect_arguments
                .as_ref()
                .map(|arguments| {
                    let limits = key
                        .ngg_subgroup_limits
                        .ok_or("NGG mesh draw is missing subgroup sizing registers")?;
                    let maximum_input_count = if draw.indexed {
                        draw.index_buffer_count
                    } else {
                        // Non-indexed indirect records carry no resource-side maximum. Reserve a
                        // practical full scene-draw envelope while retaining a packet-time count
                        // above it when the guest already exposed one.
                        draw.index_count.max(65_536)
                    };
                    self.prepare_indirect_ngg(
                        &device,
                        arguments,
                        indirect_index_buffer.as_ref().map(|(buffer, _)| buffer),
                        draw.indexed,
                        draw.index_size,
                        draw.primitive_type,
                        maximum_input_count,
                        limits,
                    )
                })
                .transpose()?
        } else {
            None
        };
        let (export_sampler_buffer, export_samplers) = self.cached_sampler_argument_buffer(
            &device,
            &vertex_function,
            vertex_sampler_buffer_index,
            export_evaluation,
        )?;
        let (pixel_sampler_buffer, pixel_samplers) = self.cached_sampler_argument_buffer(
            &device,
            &pixel_function,
            pixel_sampler_buffer_index,
            pixel_evaluation,
        )?;
        if !vertex_heap_texture_tables.is_empty() {
            return Err("vertex-stage heap texture tables are not yet encodable".to_owned());
        }
        let pixel_heap_tables = self.build_pixel_heap_texture_tables(
            &device,
            &pixel_function,
            pixel_evaluation,
            &pixel_heap_texture_tables,
        )?;
        let export_global_argument_buffer = create_global_argument_buffer(
            &device,
            &vertex_function,
            vertex_global_argument_buffer_index,
            &export_buffers,
        )?;
        let pixel_global_argument_buffer = create_global_argument_buffer(
            &device,
            &pixel_function,
            pixel_global_argument_buffer_index,
            &pixel_buffers,
        )?;
        let mut primitive = metal_primitive_type(draw.primitive_type)?;
        if uses_procedural_rectangle_stage {
            // The synthesized covering uses a single oversized triangle; guest strip/rect
            // topology and base vertex must not reshape that draw.
            primitive = MTLPrimitiveType::Triangle;
        }
        // Cover draws must not consume guest index buffers: an indexed cover would take the
        // draw_indexed path below and ignore the synthesized three-vertex triangle.
        let indexed_draw = if uses_procedural_rectangle_stage {
            None
        } else if draw.indexed && indirect_arguments.is_none() {
            let indices = read_draw_indices(draw).ok_or("index buffer is unreadable")?;
            let prepared = prepare_indexed_primitive(draw, indices)?;
            primitive = prepared.1;
            let indices = prepared.0;
            let buffer = self.cached_index_buffer(&device, indices.as_ref())?;
            Some((indices, buffer))
        } else {
            None
        };
        let ngg_indices = if uses_mesh_stage && indirect_ngg.is_none() {
            if let Some((indices, _)) = indexed_draw.as_ref() {
                Some(prepare_indexed_ngg_indices(
                    draw,
                    Arc::clone(indices),
                    primitive,
                )?)
            } else if !draw.indexed && matches!(draw.primitive_type, 1 | 6) {
                Some(prepare_nonindexed_ngg_indices(draw)?)
            } else {
                return Err("NGG mesh draw has an unsupported source topology".to_owned());
            }
        } else {
            None
        };
        let ngg_meshlets = if let Some(indices) = ngg_indices.as_ref() {
            let limits = key
                .ngg_subgroup_limits
                .ok_or("NGG mesh draw is missing subgroup sizing registers")?;
            self.cached_ngg_meshlets(&device, indices.as_ref(), limits)?
        } else {
            None
        };
        let scissor = decode_scissor(draw, render_width, render_height)?;
        let diagnostic_force_triangle = diagnostic_disable_combined_raster
            && diagnostic_complex_msaa_raster
            && diagnostic_force_msaa_vertex_pattern();
        let viewport = if diagnostic_force_triangle {
            MTLViewport {
                originX: 0.0,
                originY: 0.0,
                width: f64::from(render_width),
                height: f64::from(render_height),
                znear: 0.0,
                zfar: 1.0,
            }
        } else {
            decode_viewport(draw, render_width, render_height, scissor)
        };
        if trace_image_descriptors() && evaluation_samples_large_linear_video(pixel_evaluation) {
            static LOGGED_VIDEO_DRAWS: AtomicUsize = AtomicUsize::new(0);
            if LOGGED_VIDEO_DRAWS.fetch_add(1, Ordering::Relaxed) < 16 {
                let indices = indexed_draw.as_ref().map(|(indices, _)| indices.as_ref());
                eprintln!(
                    "[REMu][METAL][VIDEO] draw={} es={:#x} ps={:#x} indexed={} index_size={} index_offset={} index_buffer={:#x} indices={indices:?} count={} primitive={:#x} targets={targets:?} depth={depth:?} raster={raster:?} scissor={scissor:?} viewport=({:.1},{:.1} {:.1}x{:.1} {:.3}..{:.3}) depth_control={:#010x} z_info={:#010x} depth_size={:#010x} raster_control={:#010x}",
                    draw.sequence,
                    draw.export_shader_address,
                    draw.pixel_shader_address,
                    draw.indexed,
                    draw.index_size,
                    draw.index_offset,
                    draw.index_buffer_address,
                    draw.index_count,
                    draw.primitive_type,
                    viewport.originX,
                    viewport.originY,
                    viewport.width,
                    viewport.height,
                    viewport.znear,
                    viewport.zfar,
                    register_value(&draw.context_registers, DB_DEPTH_CONTROL).unwrap_or_default(),
                    register_value(&draw.context_registers, DB_Z_INFO).unwrap_or_default(),
                    register_value(&draw.context_registers, DB_DEPTH_SIZE_XY).unwrap_or_default(),
                    register_value(&draw.context_registers, PA_SU_SC_MODE_CNTL)
                        .unwrap_or(DEFAULT_PA_SU_SC_MODE_CNTL),
                );
                let global_bindings = export_evaluation
                    .global_memory_bindings
                    .iter()
                    .map(|binding| {
                        (
                            binding.scalar_address,
                            binding.base_address,
                            binding.byte_count,
                            binding.instruction_pcs.as_slice(),
                        )
                    })
                    .collect::<Vec<_>>();
                eprintln!(
                    "[REMu][METAL][VIDEO] export initial_scalars={} globals={global_bindings:x?} vertex_inputs={:?}",
                    export_evaluation.initial_scalar_registers.len(),
                    export_evaluation.vertex_inputs,
                );
            }
        }
        let export_buffer_indices = (0..export_buffers.len())
            .map(|index| u64::try_from(index).map_err(|_| "vertex buffer index overflow"))
            .collect::<Result<Vec<_>, _>>()?;
        let pixel_buffer_indices = (0..pixel_buffers.len())
            .map(|index| u64::try_from(index).map_err(|_| "pixel buffer index overflow"))
            .collect::<Result<Vec<_>, _>>()?;
        let export_texture_indices = (0..export_textures.len())
            .map(|index| u64::try_from(index).map_err(|_| "vertex texture index overflow"))
            .collect::<Result<Vec<_>, _>>()?;
        let pixel_texture_indices = (0..pixel_textures.len())
            .map(|index| u64::try_from(index).map_err(|_| "pixel texture index overflow"))
            .collect::<Result<Vec<_>, _>>()?;
        let export_user_data_register = [0x08c, 0x0cc, 0x04c]
            .into_iter()
            .find(|base| {
                draw.shader_registers
                    .iter()
                    .any(|register| register.offset == base - 1)
            })
            .unwrap_or(0x0cc);
        let mut vertex_scalar_metadata = shader_scalar_metadata_for_registers(
            export_evaluation,
            &draw.shader_registers,
            export_user_data_register,
            8,
        )?;
        if diagnostic_use_draw_merged_wave_counts() && vertex_scalar_metadata.len() > 3 {
            let vertex_threads = if let Some((indices, _)) = indexed_draw.as_ref() {
                let mut unique = HashSet::with_capacity(indices.len().min(64));
                for &index in indices.iter() {
                    unique.insert(index);
                    if unique.len() == 64 {
                        break;
                    }
                }
                unique.len()
            } else {
                usize::try_from(metal_nonindexed_vertex_count(
                    draw.primitive_type,
                    draw.index_count,
                ))
                .unwrap_or(64)
                .min(64)
            };
            let element_count = indexed_draw.as_ref().map_or_else(
                || usize::try_from(draw.index_count).unwrap_or(64),
                |(indices, _)| indices.len(),
            );
            let primitive_threads = match primitive {
                MTLPrimitiveType::Point => element_count,
                MTLPrimitiveType::Line => element_count / 2,
                MTLPrimitiveType::LineStrip => element_count.saturating_sub(1),
                MTLPrimitiveType::Triangle => element_count / 3,
                MTLPrimitiveType::TriangleStrip => element_count.saturating_sub(2),
            }
            .clamp(1, 64);
            let vertex_threads = vertex_threads.clamp(1, 64);
            vertex_scalar_metadata[3] = u32::try_from(vertex_threads).expect("wave count fits u32")
                | (u32::try_from(primitive_threads).expect("wave count fits u32") << 8);
        }
        let mut pixel_scalar_metadata = shader_scalar_metadata_for_registers(
            pixel_evaluation,
            &draw.shader_registers,
            0x00c,
            0,
        )?;
        // `crates/remu-shader-metal/src/compiler/image.rs` applies the T# destination select in
        // the shader, reading it out of `remu_initial_s[scalar_resource + 3]`. This metadata is
        // therefore the only carrier between the descriptor the host decoded and the swizzle the
        // pixel shader applies. Poppy Playtime's Slate font atlas is the one traced binding whose
        // select is not the native identity (0x800 = X0 Y0 Z0 W=R); if that word ever reaches the
        // shader as zero the 0xFAC identity fallback fires, the R8 atlas samples as (r, 0, 0, 1),
        // and every glyph quad becomes a solid opaque block. Report both sides so a tofu run says
        // which one is wrong. A descriptor the guest materializes with SMEM loads legitimately
        // differs here: the translated shader replays those loads before the sample.
        if trace_glyph_atlas() {
            for binding in &pixel_evaluation.image_bindings {
                let Some(&descriptor_word) = binding.resource_descriptor.get(3) else {
                    continue;
                };
                let select = descriptor_word & 0xfff;
                if matches!(select, 0 | 0x204 | 0xfac) {
                    continue;
                }
                let seeded = usize::try_from(binding.control.scalar_resource)
                    .ok()
                    .and_then(|scalar| scalar.checked_add(3))
                    .and_then(|scalar| pixel_scalar_metadata.get(scalar).copied());
                // K1: lock-free bounded sampler (was a mutex-guarded HashSet dedup).
                static LOGGED: AtomicUsize = AtomicUsize::new(0);
                if LOGGED.fetch_add(1, Ordering::Relaxed) < 64 {
                    eprintln!(
                        "[REMu][METAL][GLYPH-SELECT] pc={:#x} scalar_resource={} descriptor_word3={descriptor_word:#010x} descriptor_select={select:#x} seeded_word3={seeded:#010x?} seeded_select={:#x?}",
                        binding.pc,
                        binding.control.scalar_resource,
                        seeded.map(|word| word & 0xfff),
                    );
                }
            }
        }
        let target_capture_requested = diagnostic_dump_target_pixel_shader().is_some()
            || diagnostic_dump_target_pixel_instruction_count().is_some()
            || diagnostic_dump_target_address().is_some()
            || diagnostic_dump_target_address_range().is_some();
        let diagnostic_capture_enabled = if diagnostic_dump_video_target_consumer() {
            static VIDEO_CAPTURE_MATCHES: AtomicUsize = AtomicUsize::new(0);
            let is_target_candidate = draw.sequence >= diagnostic_dump_target_minimum_draw()
                && !targets.is_empty()
                && video_target_binding_summary(pixel_evaluation).is_some()
                && diagnostic_dump_target_pixel_instruction_count().is_none_or(|count| {
                    draw.pixel_shader
                        .as_ref()
                        .is_some_and(|shader| shader.program.instructions.len() == count)
                });
            is_target_candidate
                && VIDEO_CAPTURE_MATCHES.fetch_add(1, Ordering::Relaxed)
                    == diagnostic_dump_target_match_index()
        } else if diagnostic_dump_video_chain_consumer() {
            diagnostic_video_chain_selected_draw().load(Ordering::Relaxed) == draw.sequence
        } else if target_capture_requested {
            diagnostic_target_draw_filter().is_some_and(|filter| {
                targets.iter().any(|target| {
                    (!filter.auto_target_skips_scanout() || !target.direct_scanout)
                        && filter.matches(
                            draw.sequence,
                            draw.pixel_shader_address,
                            target.texture.address,
                            target.texture.width,
                            target.texture.height,
                        )
                })
            })
        } else {
            true
        };
        if diagnostic_dump_target_pixel_shader() == Some(draw.pixel_shader_address) {
            static LOGGED: AtomicUsize = AtomicUsize::new(0);
            if LOGGED.fetch_add(1, Ordering::Relaxed) < 32 {
                eprintln!(
                    "[REMu][METAL][TARGET-FILTER-CHECK] draw={} ps={:#x} ready={} armed={} key_capture={}/{} targets={:?}",
                    draw.sequence,
                    draw.pixel_shader_address,
                    diagnostic_target_after_mrt_ready(),
                    diagnostic_capture_enabled,
                    key.diagnostics.capture_vertex_position,
                    key.diagnostics.capture_pixel_output,
                    targets
                        .iter()
                        .map(|target| (
                            target.texture.address,
                            target.texture.width,
                            target.texture.height,
                            target.direct_scanout,
                        ))
                        .collect::<Vec<_>>(),
                );
            }
        }
        if key.diagnostics.capture_vertex_position {
            vertex_scalar_metadata.push(u32::from(diagnostic_capture_enabled));
        }
        if key.diagnostics.capture_pixel_output {
            pixel_scalar_metadata.push(u32::from(diagnostic_capture_enabled));
        }
        let vertex_scalar_byte_count = vertex_scalar_metadata
            .len()
            .checked_mul(4)
            .and_then(|count| u64::try_from(count).ok())
            .ok_or("vertex scalar byte count overflow")?;
        let pixel_scalar_byte_count = pixel_scalar_metadata
            .len()
            .checked_mul(4)
            .and_then(|count| u64::try_from(count).ok())
            .ok_or("pixel scalar byte count overflow")?;
        let indexed_count = indexed_draw
            .as_ref()
            .map(|(indices, _)| u64::try_from(indices.len()).map_err(|_| "index count overflow"))
            .transpose()?;
        let mut pending_writebacks = Vec::new();
        collect_pending_global_writebacks(
            &export_buffers,
            export_evaluation,
            &mut pending_writebacks,
        )?;
        collect_pending_global_writebacks(
            &pixel_buffers,
            pixel_evaluation,
            &mut pending_writebacks,
        )?;
        let mut pending_guest_write_ranges = Vec::new();
        collect_pending_guest_write_ranges(
            &export_buffers,
            export_evaluation,
            &mut pending_guest_write_ranges,
        )?;
        collect_pending_guest_write_ranges(
            &pixel_buffers,
            pixel_evaluation,
            &mut pending_guest_write_ranges,
        )?;
        let stencil_references = depth.as_ref().map(|target| {
            (
                target.stencil.enabled,
                target.stencil.front_reference,
                target.stencil.back_reference,
            )
        });
        let depth_stencil_state = depth
            .as_ref()
            .copied()
            .map(|depth| self.cached_depth_stencil_state(&device, depth));
        let diagnostic_vertex_position_buffer = if key.diagnostics.capture_vertex_position {
            let byte_count = DIAGNOSTIC_VERTEX_POSITION_VERTEX_CAPACITY
                .checked_mul(DIAGNOSTIC_VERTEX_STATE_WORDS_PER_VERTEX)
                .and_then(|words| words.checked_mul(std::mem::size_of::<u32>()))
                .and_then(|bytes| u64::try_from(bytes).ok())
                .ok_or("diagnostic vertex-position byte count overflow")?;
            if self.diagnostic_vertex_position_buffer.is_none() {
                let buffer = device.new_buffer(byte_count, MTLResourceOptions::StorageModeShared);
                buffer.set_label("REMu diagnostic vertex positions");
                self.diagnostic_vertex_position_buffer = Some(buffer);
            }
            // Zero only for the armed draw. The shader pair is instrumented before the semantic
            // gate opens, but its capture branch is disabled on earlier composites.
            let buffer = self
                .diagnostic_vertex_position_buffer
                .as_ref()
                .expect("diagnostic vertex-position buffer was just ensured");
            let byte_count = usize::try_from(byte_count)
                .expect("diagnostic vertex-position byte count fits usize");
            if diagnostic_capture_enabled {
                assert!(
                    super::MetalBytes::new(buffer).fill(0, 0, byte_count),
                    "diagnostic vertex-position buffer must expose byte_count shared bytes"
                );
            }
            Some(buffer.clone())
        } else {
            None
        };
        let diagnostic_pixel_output_buffer = if key.diagnostics.capture_pixel_output {
            let byte_count = DIAGNOSTIC_PIXEL_OUTPUT_WIDTH
                .checked_mul(DIAGNOSTIC_PIXEL_OUTPUT_HEIGHT)
                .and_then(|pixels| pixels.checked_mul(DIAGNOSTIC_PIXEL_OUTPUT_WORDS_PER_PIXEL))
                .and_then(|words| words.checked_mul(std::mem::size_of::<u32>()))
                .and_then(|bytes| u64::try_from(bytes).ok())
                .ok_or("diagnostic pixel-output byte count overflow")?;
            if self.diagnostic_pixel_output_buffer.is_none() {
                let buffer = device.new_buffer(byte_count, MTLResourceOptions::StorageModeShared);
                buffer.set_label("REMu diagnostic pixel outputs");
                self.diagnostic_pixel_output_buffer = Some(buffer);
            }
            // Zero only for the armed draw: without this, covered=2073600 can inherit an earlier
            // capture and cannot be attributed to the dumped TARGET draw.
            let buffer = self
                .diagnostic_pixel_output_buffer
                .as_ref()
                .expect("diagnostic pixel-output buffer was just ensured");
            let byte_count =
                usize::try_from(byte_count).expect("diagnostic pixel-output byte count fits usize");
            if diagnostic_capture_enabled {
                assert!(
                    super::MetalBytes::new(buffer).fill(0, 0, byte_count),
                    "diagnostic pixel-output buffer must expose byte_count shared bytes"
                );
            }
            Some(buffer.clone())
        } else {
            None
        };

        // HTILE is logical compression metadata, not ordinary depth pixels. New metadata is
        // cleared by definition and a guest DMA/compute fill rearms it. Consume that state once
        // at attachment use and expand it through Metal's depth load clear.
        let depth_metadata_clear = depth.is_some_and(|depth| {
            consume_depth_metadata_clear(&mut self.depth_metadata_clears, depth.htile_address)
        });
        let depth_metadata_clear_sequence = depth_metadata_clear.then(|| {
            let sequence = self.next_depth_metadata_clear_sequence;
            self.next_depth_metadata_clear_sequence =
                self.next_depth_metadata_clear_sequence.wrapping_add(1);
            sequence
        });
        let depth_clear_value = depth.and_then(|depth| {
            if depth.clear_enabled || depth_metadata_clear {
                Some(depth.clear_depth)
            } else if clear_uninitialized_depth {
                Some(initial_depth_clear_for_target(depth))
            } else {
                None
            }
        });
        let render_pass_key = GuestRenderPassKey {
            targets: targets
                .iter()
                .map(|target| {
                    (
                        target.texture,
                        target.sample_count,
                        target.direct_scanout,
                        target.texture_type,
                        target.layer_count,
                        target.base_layer,
                        target.mip_level,
                        target.mip_level_count,
                    )
                })
                .collect(),
            // An explicit DB clear is an operation, not persistent attachment state. Include it
            // in the key so it always terminates a preceding load pass, and so the following draw
            // opens a new load pass instead of continuing to clear on the same encoder.
            depth: depth.map(|depth| {
                (
                    depth.texture,
                    depth.sample_count,
                    depth.layer_count,
                    depth_clear_value.map(|value| (value as f32).to_bits()),
                    depth_metadata_clear_sequence,
                )
            }),
        };
        let needs_new_render_pass = self
            .active_open_guest_batch()
            .and_then(|batch| batch.render_pass.as_ref())
            .is_none_or(|open| open.key != render_pass_key);
        let pass = if needs_new_render_pass {
            let pass = RenderPassDescriptor::new();
            pass.set_render_target_width(u64::from(render_width));
            pass.set_render_target_height(u64::from(render_height));
            pass.set_render_target_array_length(u64::from(key.render_target_array_length));
            for (location, ((texture, resolve_texture), initial_clear_color)) in target_textures
                .iter()
                .zip(&target_resolve_textures)
                .zip(&initial_target_clear_colors)
                .enumerate()
            {
                let attachment = pass
                    .color_attachments()
                    .object_at(u64::try_from(location).map_err(|_| "color location overflow")?)
                    .ok_or_else(|| {
                        format!("Metal render pass has no color attachment {location}")
                    })?;
                attachment.set_texture(Some(texture));
                attachment.set_level(u64::from(targets[location].mip_level));
                if targets[location].texture_type == 10 {
                    attachment.set_depth_plane(u64::from(targets[location].base_layer));
                } else {
                    attachment.set_slice(u64::from(targets[location].base_layer));
                }
                if let Some(clear_color) = initial_clear_color {
                    attachment.set_load_action(MTLLoadAction::Clear);
                    attachment.set_clear_color(*clear_color);
                } else {
                    attachment.set_load_action(MTLLoadAction::Load);
                }
                // B4: resolve store action via attachment::tbdr (memoryless only when draws/pass
                // gate is proven; otherwise StoreAndMultisampleResolve — loud refusal in tbdr).
                let memoryless = super::attachment::msaa_memoryless_decision(
                    self.statistics.encoded_draws.saturating_add(1),
                    self.statistics.encoded_render_passes.saturating_add(1),
                    false,
                );
                let store =
                    super::attachment::color_store_action(resolve_texture.is_some(), &memoryless);
                if let Some(resolve_texture) = resolve_texture {
                    attachment.set_resolve_texture(Some(resolve_texture));
                }
                attachment.set_store_action(store.metal());
            }
            if let (Some(_depth), Some(texture)) = (depth, depth_texture.as_ref()) {
                let attachment = pass
                    .depth_attachment()
                    .ok_or("Metal render pass has no depth attachment")?;
                attachment.set_texture(Some(texture));
                if depth.is_some_and(|depth| depth.layer_count > 1) {
                    // `depth.texture.address` already points at DB_DEPTH_VIEW's first selected
                    // slice, so the Metal view begins at slice zero and the render-pass array
                    // length fans the draw out over the remaining selected slices.
                    attachment.set_slice(0);
                }
                if let Some(clear_depth) = depth_clear_value {
                    attachment.set_load_action(MTLLoadAction::Clear);
                    attachment.set_clear_depth(clear_depth);
                } else {
                    attachment.set_load_action(MTLLoadAction::Load);
                }
                // B4: consumer-derived depth store. Lookahead for "sampled later" is not yet
                // proven in-pass, so UnknownKeepStore retains Store (loud seam in NOTES).
                if let Some(resolve_texture) = depth_resolve_texture.as_ref() {
                    attachment.set_resolve_texture(Some(resolve_texture));
                }
                let depth_store = super::attachment::depth_store_action(
                    depth_resolve_texture.is_some(),
                    super::attachment::DepthConsumerHint::UnknownKeepStore,
                );
                attachment.set_store_action(depth_store.metal());
            }
            // J1: attach visibility result buffer so Counting mode can land sample counts.
            let visibility = super::attachment::ensure_visibility_result_buffer(
                &device,
                &mut self.visibility_result_buffer,
            );
            super::attachment::attach_visibility_result_buffer(pass, &visibility);
            Some(pass)
        } else {
            None
        };
        let ack_metal_encoding_started = ack_phase_timing.then(Instant::now);
        // J1: mark counting armed before borrowing the open batch (see arm_visibility_counting below).
        self.visibility_counting_armed = true;
        {
            let (batch, statistics) = self.active_open_guest_batch_and_statistics();
            let batch = batch.expect("guest command batch was opened above");
            batch.work.push(GuestBatchWork::Draw {
                sequence: draw.sequence,
                indexed: draw.indexed,
                indirect_arguments_address: draw.indirect_arguments_address,
                export: key.export_shader_address,
                pixel: key.pixel_shader_address,
                index_count: draw.index_count,
                instance_count: draw.instance_count,
                target: draw.color_target_address,
                extent: [render_width, render_height],
                export_msl_bytes: vertex_msl_bytes,
                pixel_msl_bytes,
                export_dispatcher: vertex_dispatcher,
                pixel_dispatcher,
            });
            append_unique_global_writebacks(&mut batch.global_writebacks, pending_writebacks);
            append_unique_guest_write_ranges(
                &mut batch.guest_write_ranges,
                pending_guest_write_ranges,
            );
            if needs_new_render_pass {
                let key_transition = batch
                    .render_pass
                    .as_ref()
                    .map(|open| open.key.transition_to(&render_pass_key));
                batch.end_compute_pass();
                if batch.end_render_pass() {
                    statistics.render_pass_ends_for_target_change = statistics
                        .render_pass_ends_for_target_change
                        .saturating_add(1);
                    statistics
                        .render_pass_key_transitions
                        .record(key_transition.expect("an ended render pass had a key"));
                }
                let encoder = batch
                    .command_buffer
                    .new_render_command_encoder(
                        pass.expect("new guest render pass descriptor was prepared"),
                    )
                    .to_owned();
                encoder.set_label("REMu translated guest render pass");
                batch.render_pass = Some(OpenGuestRenderPass {
                    key: render_pass_key,
                    encoder,
                });
                statistics.encoded_render_passes =
                    statistics.encoded_render_passes.saturating_add(1);
            }
            let encoder = &batch
                .render_pass
                .as_ref()
                .expect("compatible guest render pass was opened above")
                .encoder;
            // J1: every translated draw contributes to the next PIXEL_PIPE_STAT_DUMP / OcclusionQuery.
            // Flag is set before the batch borrow so OcclusionQuery can observe it without fighting
            // the open-batch mutable borrow.
            super::attachment::arm_visibility_counting(encoder);
            encoder.set_render_pipeline_state(&pipeline);
            encoder.set_front_facing_winding(raster.front_winding);
            encoder.set_cull_mode(
                if diagnostic_disable_culling_for_draw || uses_procedural_rectangle_stage {
                    MTLCullMode::None
                } else {
                    raster.cull_mode
                },
            );
            encoder.set_triangle_fill_mode(raster.fill_mode);
            if !diagnostic_disable_depth_for_draw
                && let Some(depth_stencil_state) = depth_stencil_state.as_ref()
            {
                encoder.set_depth_stencil_state(depth_stencil_state);
                if let Some((true, front_reference, back_reference)) = stencil_references {
                    encoder.set_stencil_front_back_reference_value(front_reference, back_reference);
                }
            }
            let bias = raster.depth_bias;
            if bias.slope_scale != 0.0 || bias.constant != 0.0 || bias.clamp != 0.0 {
                encoder.set_depth_bias(bias.constant, bias.slope_scale, bias.clamp);
            }
            encoder.set_blend_color(
                f32::from_bits(
                    register_value(&draw.context_registers, CB_BLEND_RED).unwrap_or_default(),
                ),
                f32::from_bits(
                    register_value(&draw.context_registers, CB_BLEND_GREEN).unwrap_or_default(),
                ),
                f32::from_bits(
                    register_value(&draw.context_registers, CB_BLEND_BLUE).unwrap_or_default(),
                ),
                f32::from_bits(
                    register_value(&draw.context_registers, CB_BLEND_ALPHA).unwrap_or_default(),
                ),
            );
            encoder.set_viewport(viewport);
            if !diagnostic_force_triangle
                && let Some(rect) =
                    metal_scissor_rect_for_draw(scissor, render_width, render_height)
            {
                // Scissor state persists for the lifetime of a Metal render encoder. A full
                // guest scissor must therefore be programmed explicitly: otherwise a preceding
                // restricted draw clips this draw when their compatible render pass is reused.
                encoder.set_scissor_rect(rect);
            }
            if let (Some(index), Some(argument_buffer)) = (
                vertex_global_argument_buffer_index,
                export_global_argument_buffer.as_ref(),
            ) {
                if uses_mesh_stage {
                    encoder.set_mesh_buffer(u64::from(index), Some(argument_buffer), 0);
                } else {
                    encoder.set_vertex_buffer(u64::from(index), Some(argument_buffer), 0);
                }
                for (buffer, binding) in export_buffers
                    .iter()
                    .zip(&export_evaluation.global_memory_bindings)
                {
                    let usage = if binding.writable {
                        MTLResourceUsage::Read | MTLResourceUsage::Write
                    } else {
                        MTLResourceUsage::Read
                    };
                    encoder.use_resource_at(
                        &buffer.buffer,
                        usage,
                        export_resource_render_stage(uses_mesh_stage),
                    );
                }
            } else {
                for (index, buffer) in export_buffer_indices.iter().zip(&export_buffers) {
                    if uses_mesh_stage {
                        encoder.set_mesh_buffer(*index, Some(&buffer.buffer), buffer.offset);
                    } else {
                        encoder.set_vertex_buffer(*index, Some(&buffer.buffer), buffer.offset);
                    }
                }
            }
            if let (Some(index), Some(argument_buffer)) = (
                pixel_global_argument_buffer_index,
                pixel_global_argument_buffer.as_ref(),
            ) {
                encoder.set_fragment_buffer(u64::from(index), Some(argument_buffer), 0);
                for (buffer, binding) in pixel_buffers
                    .iter()
                    .zip(&pixel_evaluation.global_memory_bindings)
                {
                    let usage = if binding.writable {
                        MTLResourceUsage::Read | MTLResourceUsage::Write
                    } else {
                        MTLResourceUsage::Read
                    };
                    encoder.use_resource_at(&buffer.buffer, usage, MTLRenderStages::Fragment);
                }
            } else {
                for (index, buffer) in pixel_buffer_indices.iter().zip(&pixel_buffers) {
                    encoder.set_fragment_buffer(*index, Some(&buffer.buffer), buffer.offset);
                }
            }
            if uses_mesh_stage {
                encoder.set_mesh_bytes(
                    u64::from(vertex_scalar_buffer_index),
                    vertex_scalar_byte_count,
                    vertex_scalar_metadata.as_ptr().cast::<c_void>(),
                );
            } else {
                encoder.set_vertex_bytes(
                    u64::from(vertex_scalar_buffer_index),
                    vertex_scalar_byte_count,
                    vertex_scalar_metadata.as_ptr().cast::<c_void>(),
                );
            }
            if let Some(buffer) = diagnostic_vertex_position_buffer.as_ref() {
                if uses_mesh_stage {
                    encoder.set_mesh_buffer(DIAGNOSTIC_MESH_POSITION_BUFFER_INDEX, Some(buffer), 0);
                } else {
                    encoder.set_vertex_buffer(
                        DIAGNOSTIC_VERTEX_POSITION_BUFFER_INDEX,
                        Some(buffer),
                        0,
                    );
                }
            }
            encoder.set_fragment_bytes(
                u64::from(pixel_scalar_buffer_index),
                pixel_scalar_byte_count,
                pixel_scalar_metadata.as_ptr().cast::<c_void>(),
            );
            if let Some(buffer) = diagnostic_pixel_output_buffer.as_ref() {
                encoder.set_fragment_buffer(DIAGNOSTIC_PIXEL_OUTPUT_BUFFER_INDEX, Some(buffer), 0);
            }
            for (index, texture) in export_texture_indices.iter().zip(&export_textures) {
                if uses_mesh_stage {
                    encoder.set_mesh_texture(*index, Some(&texture.texture));
                } else {
                    encoder.set_vertex_texture(*index, Some(&texture.texture));
                }
            }
            for (index, texture) in pixel_texture_indices.iter().zip(&pixel_textures) {
                encoder.set_fragment_texture(*index, Some(&texture.texture));
            }
            for (buffer_slot, argument_buffer, textures) in &pixel_heap_tables {
                encoder.set_fragment_buffer(
                    u64::from(*buffer_slot),
                    Some(argument_buffer),
                    0,
                );
                for texture in textures {
                    encoder.use_resource(texture, metal::MTLResourceUsage::Read);
                }
            }
            if let (Some(index), Some(buffer)) =
                (vertex_sampler_buffer_index, export_sampler_buffer.as_ref())
            {
                if uses_mesh_stage {
                    encoder.set_mesh_buffer(u64::from(index), Some(buffer), 0);
                } else {
                    encoder.set_vertex_buffer(u64::from(index), Some(buffer), 0);
                }
            } else {
                for (index, sampler) in export_samplers.iter().enumerate() {
                    let index =
                        u64::try_from(index).map_err(|_| "vertex sampler index overflow")?;
                    if uses_mesh_stage {
                        encoder.set_mesh_sampler_state(index, Some(sampler));
                    } else {
                        encoder.set_vertex_sampler_state(index, Some(sampler));
                    }
                }
            }
            if let (Some(index), Some(buffer)) =
                (pixel_sampler_buffer_index, pixel_sampler_buffer.as_ref())
            {
                encoder.set_fragment_buffer(u64::from(index), Some(buffer), 0);
            } else {
                for (index, sampler) in pixel_samplers.iter().enumerate() {
                    encoder.set_fragment_sampler_state(
                        u64::try_from(index).map_err(|_| "pixel sampler index overflow")?,
                        Some(sampler),
                    );
                }
            }

            if uses_mesh_stage {
                if let Some(indirect) = indirect_ngg.as_ref() {
                    if !matches!(scissor, GuestScissor::Empty) {
                        encoder.set_mesh_buffer(
                            NGG_MESHLET_BUFFER_INDEX,
                            Some(&indirect.meshlets),
                            indirect.meshlets_offset,
                        );
                        encoder.set_mesh_buffer(
                            NGG_MESH_DRAW_BUFFER_INDEX,
                            Some(&indirect.mesh_draw),
                            indirect.mesh_draw_offset,
                        );
                        encoder.draw_mesh_threadgroups_with_indirect_buffer(
                            &indirect.mesh_dispatch,
                            indirect.mesh_dispatch_offset,
                            MTLSize::new(1, 1, 1),
                            MTLSize::new(
                                u64::from(
                                    key.ngg_subgroup_limits
                                        .expect(
                                            "NGG subgroup limits were validated before encoding",
                                        )
                                        .metal_threadgroup_threads(),
                                ),
                                1,
                                1,
                            ),
                        );
                    } else {
                        super::stats::NOOP_DRAWS_MESH_DISPATCH_SKIPPED
                            .fetch_add(1, Ordering::Relaxed);
                        super::stats::note_noop_draw("gpu:noop-mesh-skip");
                    }
                } else {
                    let indices = ngg_indices
                        .as_ref()
                        .expect("NGG indices were prepared before opening the render pass");
                    if ngg_mesh_dispatch_required(
                        scissor,
                        draw.instance_count,
                        ngg_meshlets.is_some(),
                    ) && let Some(meshlets) = ngg_meshlets.as_ref()
                    {
                        let mesh_draw = [
                            0_u32,
                            0_u32,
                            u32::from_ne_bytes(ngg_mesh_base_vertex(draw).to_ne_bytes()),
                            0_u32,
                        ];
                        encoder.set_mesh_buffer(
                            NGG_MESHLET_BUFFER_INDEX,
                            Some(&meshlets.buffer),
                            0,
                        );
                        encoder.set_mesh_bytes(
                            NGG_MESH_DRAW_BUFFER_INDEX,
                            u64::try_from(std::mem::size_of_val(&mesh_draw))
                                .expect("mesh draw metadata size fits u64"),
                            mesh_draw.as_ptr().cast::<c_void>(),
                        );
                        static LOGGED_MESH_DRAWS: AtomicUsize = AtomicUsize::new(0);
                        let mesh_sequence = LOGGED_MESH_DRAWS.fetch_add(1, Ordering::Relaxed) + 1;
                        if mesh_sequence <= 16 || mesh_sequence.is_power_of_two() {
                            let unique = {
                                let mut values = indices.iter().copied().collect::<Vec<_>>();
                                values.sort_unstable();
                                values.dedup();
                                values
                            };
                            eprintln!(
                                "[REMu][METAL][NGG-MESH] sequence={mesh_sequence} draw={} indices={} raw={:?} unique={unique:?} primitives={} packed_vertices={} instances={} groups={}x{} cull={:?} winding={:?}",
                                draw.sequence,
                                indices.len(),
                                &indices[..indices.len().min(12)],
                                meshlets.primitive_count,
                                meshlets.vertex_references,
                                draw.instance_count,
                                meshlets.group_count,
                                draw.instance_count,
                                raster.cull_mode,
                                raster.front_winding,
                            );
                        }
                        encoder.draw_mesh_threadgroups(
                            MTLSize::new(meshlets.group_count, u64::from(draw.instance_count), 1),
                            MTLSize::new(1, 1, 1),
                            MTLSize::new(
                                u64::from(
                                    key.ngg_subgroup_limits
                                        .expect(
                                            "NGG subgroup limits were validated before encoding",
                                        )
                                        .metal_threadgroup_threads(),
                                ),
                                1,
                                1,
                            ),
                        );
                    } else {
                        // A mesh-stage draw whose dispatch predicate declines encodes nothing at
                        // all; without a count that whole draw disappears without a trace.
                        super::stats::NOOP_DRAWS_MESH_DISPATCH_SKIPPED
                            .fetch_add(1, Ordering::Relaxed);
                        super::stats::note_noop_draw("gpu:noop-mesh-skip");
                    }
                }
            } else if diagnostic_force_triangle {
                encoder.draw_primitives_instanced(MTLPrimitiveType::Triangle, 0, 3, 1);
            } else if matches!(scissor, GuestScissor::Empty) {
                // Metal rejects an empty scissor, so a guest empty-scissor draw encodes no
                // primitive while preserving command-stream ordering.
                super::stats::NOOP_DRAWS_EMPTY_SCISSOR.fetch_add(1, Ordering::Relaxed);
                super::stats::note_noop_draw("gpu:noop-empty-scissor");
            } else if let Some(arguments) = indirect_arguments.as_ref() {
                if let Some((index_buffer, index_type)) = indirect_index_buffer.as_ref() {
                    encoder.draw_indexed_primitives_indirect(
                        primitive,
                        *index_type,
                        &index_buffer.buffer,
                        index_buffer.offset,
                        &arguments.buffer,
                        arguments.offset,
                    );
                } else {
                    encoder.draw_primitives_indirect(
                        primitive,
                        &arguments.buffer,
                        arguments.offset,
                    );
                }
            } else if let Some((_, index_buffer)) = indexed_draw.as_ref() {
                encoder.draw_indexed_primitives_instanced_base_instance(
                    primitive,
                    indexed_count.expect("indexed draw count was precomputed"),
                    MTLIndexType::UInt32,
                    index_buffer,
                    0,
                    u64::from(draw.instance_count),
                    i64::from(draw.base_vertex),
                    0,
                );
            } else if uses_procedural_rectangle_stage {
                static LOGGED_PROCEDURAL_RECTANGLES: AtomicUsize = AtomicUsize::new(0);
                let sequence = LOGGED_PROCEDURAL_RECTANGLES.fetch_add(1, Ordering::Relaxed) + 1;
                if sequence <= 16 || sequence.is_power_of_two() {
                    eprintln!(
                        "[REMu][METAL][PROCEDURAL-RECT] sequence={sequence} draw={} guest_count={} guest_primitive={:#x} covering=triangle3",
                        draw.sequence, draw.index_count, draw.primitive_type,
                    );
                }
                encoder.draw_primitives_instanced(
                    MTLPrimitiveType::Triangle,
                    0,
                    3,
                    u64::from(draw.instance_count),
                );
            } else {
                let first_vertex = i64::from(draw.index_offset) + i64::from(draw.base_vertex);
                encoder.draw_primitives_instanced(
                    primitive,
                    u64::try_from(first_vertex)
                        .map_err(|_| format!("negative first vertex {first_vertex}"))?,
                    u64::from(metal_nonindexed_vertex_count(
                        draw.primitive_type,
                        draw.index_count,
                    )),
                    u64::from(draw.instance_count),
                );
            }
        }
        let ack_metal_encoding_finished = ack_phase_timing.then(Instant::now);

        for (target, resident_descriptor) in targets.iter().zip(&target_resident_descriptors) {
            if target.direct_scanout {
                if let Some(batch) = self.active_open_guest_batch_mut() {
                    batch.record_resident_write_footprint(
                        SharedEventWriteKind::DirectScanoutTarget,
                        target.span,
                    );
                }
                self.mark_direct_scanout_initialized(target.texture.address);
            } else {
                let region = WrittenRegion::render_target(*target, scissor);
                if target.sample_count > 1 {
                    self.mark_multisample_texture_initialized(target.texture, target.sample_count);
                    self.mark_resident_texture_written_preserving_multisample_at(
                        target.texture,
                        target.span,
                        *resident_descriptor,
                        region,
                        activity_stamp,
                    );
                } else {
                    self.mark_resident_texture_written_at(
                        target.texture,
                        target.span,
                        *resident_descriptor,
                        region,
                        activity_stamp,
                    );
                }
            }
            // A flip can arrive before the presenter has translated its render target. Remember
            // that this submission wrote the exact target, but defer presentation until the
            // boundary so later compositor and interface passes cannot be skipped.
            self.note_pending_flip_target_write(target.texture);
        }
        diagnostic_note_full_resolution_mrt_draw(
            draw.sequence,
            targets.len(),
            !targets.is_empty()
                && targets
                    .iter()
                    .all(|target| target.texture.width >= 1_280 && target.texture.height >= 720),
        );
        let dump_target_before_pixel_shader =
            diagnostic_dump_target_before_pixel_shader().is_some() || diagnostic_dump_glyph_draw();
        let trace_scanout_writers = trace_scanout_writes();
        for target in &targets {
            if should_record_diagnostic_target_writer(
                dump_target_before_pixel_shader,
                trace_scanout_writers,
                self.is_diagnostic_scanout_target(target),
            ) {
                let writer = DiagnosticScanoutWriter {
                    draw: draw.sequence,
                    export_shader: draw.export_shader_address,
                    pixel_shader: draw.pixel_shader_address,
                    queue_id: self.active_guest_queue,
                };
                record_diagnostic_scanout_writer(target.texture.address, writer);
                if trace_scanout_writers {
                    eprintln!(
                        "[REMu][METAL][SCANOUT-WRITE] writer={writer:?} target={:?} direct={}",
                        target.texture, target.direct_scanout,
                    );
                }
            }
        }
        if let Some(depth) = depth {
            if depth.sample_count > 1 {
                self.mark_multisample_texture_initialized(depth.texture, depth.sample_count);
            }
            // Any Metal depth store must claim GPU ownership. Explicit DB clears and Z writes
            // already did; HTILE/metadata expansion and first-use clears also LoadAction::Clear
            // the attachment. Leaving those unmarked kept Depth32Float residents at gpufalse/v0
            // while an earlier R32Float sample SRD shadowed them and forced CPU snapshots
            // (Octopath 0x8063d50000: depth_native=0 forever).
            if depth.write_enabled || depth_clear_value.is_some() {
                let region = WrittenRegion::depth_target(scissor, depth_clear_value.is_some());
                if depth.sample_count > 1 {
                    self.mark_resident_texture_written_preserving_multisample_at(
                        depth.texture,
                        depth.span,
                        depth_resident_descriptor,
                        region,
                        activity_stamp,
                    );
                } else {
                    self.mark_resident_texture_written_at(
                        depth.texture,
                        depth.span,
                        depth_resident_descriptor,
                        region,
                        activity_stamp,
                    );
                }
            }

            if draw.sequence >= diagnostic_dump_target_minimum_draw() && diagnostic_depth_draw {
                static DUMPED_DEPTH: AtomicBool = AtomicBool::new(false);
                if DUMPED_DEPTH
                    .compare_exchange(false, true, Ordering::Relaxed, Ordering::Relaxed)
                    .is_ok()
                {
                    self.wait_for_guest_command_batches()?;
                    let texture = self
                        .resident_guest_textures
                        .get(&depth.texture)
                        .filter(|resident| resident.serves_reads())
                        .ok_or("diagnostic depth target has no resident Metal texture")?
                        .texture
                        .clone();
                    let bytes = read_texture_linear(&device, &queue, &texture, depth.texture)?;
                    let mut finite = 0_usize;
                    let mut nonzero = 0_usize;
                    let mut minimum = f32::INFINITY;
                    let mut maximum = f32::NEG_INFINITY;
                    let mut sum = 0.0_f64;
                    for pixel in bytes.chunks_exact(4) {
                        let value = f32::from_bits(u32::from_le_bytes([
                            pixel[0], pixel[1], pixel[2], pixel[3],
                        ]));
                        nonzero += usize::from(value.to_bits() != 0);
                        if value.is_finite() {
                            finite += 1;
                            minimum = minimum.min(value);
                            maximum = maximum.max(value);
                            sum += f64::from(value);
                        }
                    }
                    let mean = if finite == 0 {
                        0.0
                    } else {
                        sum / finite as f64
                    };
                    let vertex_positions = self
                        .diagnostic_vertex_position_buffer
                        .as_ref()
                        .and_then(|buffer| {
                            let word_count = DIAGNOSTIC_VERTEX_POSITION_VERTEX_CAPACITY
                                .checked_mul(DIAGNOSTIC_VERTEX_STATE_WORDS_PER_VERTEX)?;
                            let words =
                                super::MetalBytes::new(buffer).typed_slice::<u32>(0, word_count)?;
                            let indices = if uses_mesh_stage {
                                // Mesh diagnostics are indexed by the Metal mesh-local thread,
                                // not by the guest index buffer consumed inside that thread.
                                (0..24).collect::<Vec<_>>()
                            } else {
                                let mut seen = HashSet::new();
                                indexed_draw
                                    .as_ref()?
                                    .0
                                    .iter()
                                    .copied()
                                    .filter_map(|index| {
                                        let adjusted =
                                            i64::from(index) + i64::from(draw.base_vertex);
                                        u32::try_from(adjusted).ok()
                                    })
                                    .filter(|index| {
                                        usize::try_from(*index).is_ok_and(|index| {
                                            index < DIAGNOSTIC_VERTEX_POSITION_VERTEX_CAPACITY
                                        }) && seen.insert(*index)
                                    })
                                    .take(24)
                                    .collect()
                            };
                            Some(
                                indices
                                    .into_iter()
                                    .map(|index| {
                                        let base = usize::try_from(index)
                                            .expect("filtered diagnostic vertex index fits usize")
                                            * DIAGNOSTIC_VERTEX_STATE_WORDS_PER_VERTEX;
                                        let raw = [
                                            words[base],
                                            words[base + 1],
                                            words[base + 2],
                                            words[base + 3],
                                        ];
                                        (index, raw, raw.map(f32::from_bits))
                                    })
                                    .collect::<Vec<_>>(),
                            )
                        })
                        .unwrap_or_default();
                    let context = diagnostic_context_register_snapshot(&draw.context_registers);
                    let shader = diagnostic_shader_summary(
                        "export",
                        draw.export_shader_address,
                        draw.export_shader_prefix,
                        draw.export_shader.as_ref(),
                    );
                    eprintln!(
                        "[REMu][METAL][DEPTH-TARGET] draw={} es={:#x} indexed={} count={} instances={} primitive={:#x} target={:?} bytes={} finite={finite} nonzero={nonzero} min={minimum} max={maximum} mean={mean} fingerprint={:016x} vertices={vertex_positions:?} context={context:08x?} shader={shader:08x?} vertex_scalars={:08x?} vertex_globals_evaluated={:x?} vertex_globals_submitted={:x?}",
                        draw.sequence,
                        draw.export_shader_address,
                        draw.indexed,
                        draw.index_count,
                        draw.instance_count,
                        draw.primitive_type,
                        depth,
                        bytes.len(),
                        guest_frame_fingerprint(&bytes),
                        vertex_scalar_metadata,
                        export_evaluation
                            .global_memory_bindings
                            .iter()
                            .map(|binding| (
                                binding.scalar_address,
                                binding.base_address,
                                binding.byte_count,
                                diagnostic_global_binding_head(binding),
                            ))
                            .collect::<Vec<_>>(),
                        export_evaluation
                            .global_memory_bindings
                            .iter()
                            .zip(&export_buffers)
                            .map(|(binding, uploaded)| (
                                binding.scalar_address,
                                binding.base_address,
                                binding.byte_count,
                                binding.writable,
                                diagnostic_uploaded_global_binding_head(binding, uploaded),
                            ))
                            .collect::<Vec<_>>(),
                    );
                }
            }
        }
        let diagnostic_sampled_storage = diagnostic_texture_writer_enabled().then(|| {
            sampled_image_bindings(pixel_evaluation)
                .filter_map(|binding| decode_image_descriptor(&binding.resource_descriptor).ok())
                .map(|descriptor| descriptor.guest)
                .collect::<Vec<_>>()
        });
        for (stage, shader, textures) in [
            (
                "vertex-storage",
                draw.export_shader_address,
                export_textures.as_slice(),
            ),
            (
                "pixel-storage",
                draw.pixel_shader_address,
                pixel_textures.as_slice(),
            ),
        ] {
            for texture in textures.iter().filter(|texture| texture.writable) {
                if let Some(sampled) = diagnostic_sampled_storage.as_ref() {
                    record_diagnostic_texture_writer(
                        texture.descriptor.guest,
                        stage,
                        draw.sequence,
                        shader,
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
                self.note_pending_flip_target_write(texture.descriptor.guest);
            }
        }
        if let (Some(slot), Some(target)) =
            (diagnostic_scanout_writer_slot, diagnostic_scanout_target)
        {
            self.capture_scanout_writer_after_draw(slot, draw, target)?;
        }
        let diagnostic_target_filter = diagnostic_target_draw_filter();
        if diagnostic_dump_target_pixel_shader() == Some(draw.pixel_shader_address)
            && diagnostic_target_after_mrt_ready()
        {
            static LOGGED_READY_TARGET_FILTERS: AtomicUsize = AtomicUsize::new(0);
            if LOGGED_READY_TARGET_FILTERS.fetch_add(1, Ordering::Relaxed) < 8 {
                let sampled_addresses = diagnostic_sampled_storage
                    .iter()
                    .flatten()
                    .map(|descriptor| descriptor.address)
                    .collect::<Vec<_>>();
                let target_matches = diagnostic_target_filter.map(|filter| {
                    targets
                        .iter()
                        .map(|target| {
                            (
                                target.texture.address,
                                target.texture.width,
                                target.texture.height,
                                target.direct_scanout,
                                filter.matches(
                                    draw.sequence,
                                    draw.pixel_shader_address,
                                    target.texture.address,
                                    target.texture.width,
                                    target.texture.height,
                                ),
                            )
                        })
                        .collect::<Vec<_>>()
                });
                eprintln!(
                    "[REMu][METAL][TARGET-FILTER-READY] draw={} ps={:#x} filter={diagnostic_target_filter:?} sampled={sampled_addresses:#x?} targets={target_matches:?}",
                    draw.sequence, draw.pixel_shader_address,
                );
            }
        }
        let diagnostic_target = diagnostic_target_filter
            .and_then(|filter| {
                if !filter.matches_sampled_addresses(
                    diagnostic_sampled_storage
                        .iter()
                        .flatten()
                        .map(|descriptor| descriptor.address),
                ) {
                    return None;
                }
                targets.iter().find(|target| {
                    // ADDRESS=auto historically matched the first 1920x1080 target, which for the
                    // Silent Hill title composite is often the early scanout cover written by the
                    // same PS before the non-scanout intermediate. Prefer non-scanout when the
                    // filter did not name an exact address, so occupancy dumps land on the
                    // intermediate the compositor actually samples.
                    if filter.auto_target_skips_scanout() && target.direct_scanout {
                        return false;
                    }
                    filter.matches(
                        draw.sequence,
                        draw.pixel_shader_address,
                        target.texture.address,
                        target.texture.width,
                        target.texture.height,
                    )
                })
            })
            .or_else(|| {
                diagnostic_dump_target_pixel_instruction_count()
                    .filter(|&count| {
                        draw.pixel_shader
                            .as_ref()
                            .is_some_and(|shader| shader.program.instructions.len() == count)
                            && (!diagnostic_dump_video_target_consumer()
                                || video_target_binding_summary(pixel_evaluation).is_some())
                    })
                    .and_then(|_| targets.first())
            })
            .or_else(|| {
                (diagnostic_dump_complex_msaa()
                    && draw
                        .pixel_shader
                        .as_ref()
                        .is_some_and(|shader| shader.program.instructions.len() == 370))
                .then(|| targets.iter().find(|target| target.sample_count > 1))
                .flatten()
            })
            .or_else(|| diagnostic_gbuffer_draw.then(|| targets.get(4)).flatten())
            .or_else(|| {
                // Title BasePass MRT0 remaps across runs (8068e0 ↔ 8068f4). Select by decoded
                // full-res attachment count instead of a brittle absolute VA.
                diagnostic_dump_target_mrt_min().and_then(|minimum| {
                    (targets.len() >= minimum
                        && targets.iter().all(|target| {
                            target.texture.width >= 1_280 && target.texture.height >= 720
                        }))
                    .then(|| targets.first())
                    .flatten()
                })
            })
            .or_else(|| {
                (diagnostic_dump_shadow_consumer()
                    && evaluation_samples_layered_comparison(pixel_evaluation))
                .then(|| targets.first())
                .flatten()
            })
            .or_else(|| {
                (diagnostic_dump_video_draw()
                    && evaluation_samples_large_linear_video(pixel_evaluation))
                .then(|| targets.first())
                .flatten()
            })
            .or_else(|| {
                (diagnostic_dump_video_target_consumer()
                    && video_target_binding_summary(pixel_evaluation).is_some())
                .then(|| targets.first())
                .flatten()
            })
            .or_else(|| {
                (diagnostic_dump_video_chain_consumer()
                    && diagnostic_video_chain_selected_draw().load(Ordering::Relaxed)
                        == draw.sequence)
                    .then(|| targets.first())
                    .flatten()
            })
            .or_else(|| {
                (diagnostic_dump_glyph_draw() && evaluation_samples_glyph_atlas(pixel_evaluation))
                    .then(|| targets.first())
                    .flatten()
            });
        if draw.sequence >= diagnostic_dump_target_minimum_draw()
            && diagnostic_dump_target_index_count().is_none_or(|count| count == draw.index_count)
            && let Some(target) = diagnostic_target
        {
            static TARGET_MATCHES: AtomicUsize = AtomicUsize::new(0);
            let target_match = TARGET_MATCHES.fetch_add(1, Ordering::Relaxed);
            if diagnostic_dump_shadow_consumer()
                && (target_match < 32 || target_match.is_multiple_of(128))
            {
                eprintln!(
                    "[REMu][METAL][SHADOW-CONSUMER] index={target_match} draw={} ps={:#x} count={} target={:#x} pipeline_targets={:?} depth={depth:?} depth_control={:#010x} z_info={:#010x}",
                    draw.sequence,
                    draw.pixel_shader_address,
                    draw.index_count,
                    target.texture.address,
                    key.targets,
                    register_value(&draw.context_registers, DB_DEPTH_CONTROL).unwrap_or_default(),
                    register_value(&draw.context_registers, DB_Z_INFO).unwrap_or_default(),
                );
                if target_match == 0 {
                    let image_descriptors = pixel_evaluation
                        .image_bindings
                        .iter()
                        .map(|binding| {
                            (
                                binding.pc,
                                binding.opcode,
                                decode_image_descriptor(&binding.resource_descriptor),
                                binding.sampler_descriptor.as_slice(),
                            )
                        })
                        .collect::<Vec<_>>();
                    eprintln!("[REMu][METAL][SHADOW-CONSUMER-IMAGES] {image_descriptors:08x?}");
                }
            }
            static DUMPED: AtomicBool = AtomicBool::new(false);
            if target_match == diagnostic_dump_target_match_index()
                && DUMPED
                    .compare_exchange(false, true, Ordering::Relaxed, Ordering::Relaxed)
                    .is_ok()
            {
                let context_registers =
                    diagnostic_context_register_snapshot(&draw.context_registers);
                let shaders = [
                    diagnostic_shader_summary(
                        "export",
                        draw.export_shader_address,
                        draw.export_shader_prefix,
                        draw.export_shader.as_ref(),
                    ),
                    diagnostic_shader_summary(
                        "pixel",
                        draw.pixel_shader_address,
                        draw.pixel_shader_prefix,
                        draw.pixel_shader.as_ref(),
                    ),
                ];
                eprintln!(
                    "[REMu][METAL][TARGET-MATCH] index={target_match} draw={} es={:#x} ps={:#x} target={:?}",
                    draw.sequence, draw.export_shader_address, draw.pixel_shader_address, target,
                );
                eprintln!(
                    "[REMu][METAL][TARGET-DRAW-STATE] draw={} indexed={} count={} instances={} primitive={:#x} color_target={:#x} scanout_target={:?} context={context_registers:08x?} shaders={shaders:08x?} shader_registers={:08x?} user_config_registers={:08x?} vs_out={:#010x} resolved_ps_inputs={:08x?} hooks=(pre_target={},post_target=true,vertex_output={},pixel_output={},sample_textures={})",
                    draw.sequence,
                    draw.indexed,
                    draw.index_count,
                    draw.instance_count,
                    draw.primitive_type,
                    draw.color_target_address,
                    draw.scanout_target,
                    draw.shader_registers,
                    draw.user_config_registers,
                    key.vertex_output_control,
                    key.pixel_input_controls,
                    diagnostic_dump_target_before_pixel_shader().is_some()
                        || diagnostic_dump_glyph_draw(),
                    diagnostic_capture_vertex_position(),
                    diagnostic_capture_pixel_output(),
                    capture_target_sample_textures(),
                );
                if diagnostic_gbuffer_draw {
                    self.wait_for_active_guest_queue_command_batches()?;
                } else {
                    self.wait_for_guest_command_batches()?;
                }
                let descriptor = target.texture;
                let texture = if target.direct_scanout {
                    self.direct_scanout_textures
                        .get(&descriptor.address)
                        .ok_or("diagnostic target has no direct scanout Metal texture")?
                        .texture
                        .clone()
                } else if let Some(resident) = self
                    .resident_guest_textures
                    .get(&descriptor)
                    .filter(|resident| resident.serves_reads())
                {
                    resident.texture.clone()
                } else if let Some(snapshot) =
                    self.scanout_snapshot_textures.get(&descriptor.address)
                {
                    snapshot.texture.clone()
                } else {
                    self.resident_guest_textures
                        .values()
                        .find(|resident| {
                            resident.serves_reads()
                                && guest_texture_descriptors_share_storage(
                                    resident.descriptor,
                                    descriptor,
                                )
                        })
                        .ok_or("diagnostic target has no resident or scanout Metal texture")?
                        .texture
                        .clone()
                };
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
                // The title compositor's exposure is loaded by its export/mesh shader, while the
                // ordinary target dump below only inventories pixel-stage textures. Preserve the
                // exact native bytes bound for EyeAdaptation-class inputs so a zero guest shadow,
                // stale history resident, or correct Metal value can be distinguished in one run.
                for (slot, (binding, uploaded)) in canonical_texture_bindings(export_evaluation)
                    .into_iter()
                    .zip(&export_textures)
                    .enumerate()
                    .filter(|(_, (_, uploaded))| {
                        uploaded.descriptor.guest.width == 1
                            && uploaded.descriptor.guest.height == 1
                            && uploaded.descriptor.guest.pixel_format == MTLPixelFormat::RGBA32Float
                    })
                {
                    let readback_descriptor = GuestTextureDescriptor {
                        pixel_format: uploaded.texture.pixel_format(),
                        bytes_per_pixel: metal_uncompressed_bytes_per_pixel(
                            uploaded.texture.pixel_format(),
                        )
                        .unwrap_or(uploaded.descriptor.guest.bytes_per_pixel),
                        ..uploaded.descriptor.guest
                    };
                    let native = read_texture_linear(
                        &device,
                        &queue,
                        &uploaded.texture,
                        readback_descriptor,
                    )
                    .map(|bytes| {
                        (
                            bytes.iter().filter(|&&value| value != 0).count(),
                            guest_frame_fingerprint(&bytes),
                            bytes.get(..bytes.len().min(32)).unwrap_or(&[]).to_vec(),
                        )
                    });
                    let guest =
                        snapshot_guest_texture_raw(uploaded.descriptor.guest).map(|bytes| {
                            (
                                bytes.iter().filter(|&&value| value != 0).count(),
                                guest_frame_fingerprint(&bytes),
                                bytes.get(..bytes.len().min(32)).unwrap_or(&[]).to_vec(),
                            )
                        });
                    let selected = uploaded.selected_resident.and_then(|descriptor| {
                        self.resident_guest_textures
                            .get(&descriptor)
                            .map(|resident| {
                                (
                                    descriptor,
                                    resident.state,
                                    resident.serves_reads(),
                                    resident.gpu_written(),
                                    resident.content_version,
                                    resident.owned.byte_count(),
                                    resident.gpu_authored.byte_count(),
                                    resident.last_use,
                                )
                            })
                    });
                    eprintln!(
                        "[REMu][METAL][TARGET-EXPORT-TEXTURE] draw={} slot={slot} pc={:#x} op={} words={:08x?} requested={:?} selected={selected:?} native_format={:?} native={native:x?} guest={guest:x?}",
                        draw.sequence,
                        binding.pc,
                        binding.opcode,
                        binding.resource_descriptor,
                        uploaded.descriptor,
                        uploaded.texture.pixel_format(),
                    );
                }
                if diagnostic_gbuffer_draw {
                    let mut attachment_summaries = Vec::with_capacity(targets.len());
                    for attachment_target in &targets {
                        let attachment_descriptor = attachment_target.texture;
                        let attachment_texture = if attachment_target.direct_scanout {
                            self.direct_scanout_textures
                                .get(&attachment_descriptor.address)
                                .ok_or("diagnostic G-buffer target has no direct scanout texture")?
                                .texture
                                .clone()
                        } else {
                            self.resident_guest_textures
                                .get(&attachment_descriptor)
                                .filter(|resident| resident.serves_reads())
                                .ok_or("diagnostic G-buffer target has no resident Metal texture")?
                                .texture
                                .clone()
                        };
                        let attachment_bytes = read_texture_linear(
                            &device,
                            &queue,
                            &attachment_texture,
                            attachment_descriptor,
                        )?;
                        let attachment_fingerprint = guest_frame_fingerprint(&attachment_bytes);
                        let attachment_nonzero =
                            attachment_bytes.iter().filter(|&&value| value != 0).count();
                        let attachment_nonzero_pixels = attachment_bytes
                            .chunks_exact(attachment_descriptor.bytes_per_pixel as usize)
                            .filter(|pixel| pixel.iter().any(|&value| value != 0))
                            .count();
                        let capture = self.capture_directory.as_deref().and_then(|directory| {
                            native_texture_capture_to_bgra(&attachment_bytes, attachment_descriptor)
                                .and_then(|bgra| {
                                    capture_guest_frame(
                                        directory,
                                        &bgra,
                                        attachment_descriptor,
                                        attachment_fingerprint,
                                    )
                                    .ok()
                                })
                        });
                        attachment_summaries.push((
                            attachment_target.guest_slot,
                            attachment_descriptor,
                            attachment_target.output_kind,
                            attachment_target.write_mask,
                            attachment_bytes.len(),
                            attachment_nonzero,
                            attachment_nonzero_pixels,
                            attachment_fingerprint,
                            attachment_bytes
                                .get(..attachment_bytes.len().min(32))
                                .unwrap_or(&[])
                                .to_vec(),
                            capture,
                        ));
                    }
                    eprintln!(
                        "[REMu][METAL][GBUFFER-TARGETS] draw={} es={:#x} ps={:#x} attachments={attachment_summaries:02x?}",
                        draw.sequence, draw.export_shader_address, draw.pixel_shader_address,
                    );
                }
                let texture_summaries = canonical_texture_bindings(pixel_evaluation)
                    .into_iter()
                    .zip(&pixel_textures)
                    .take(4)
                    .enumerate()
                    .map(|(slot, (binding, uploaded))| {
                        // Keep the guest descriptor and the exact native view together. Component
                        // reordered aliases intentionally have different guest/native formats.
                        let readback_descriptor = GuestTextureDescriptor {
                            pixel_format: uploaded.texture.pixel_format(),
                            bytes_per_pixel: metal_uncompressed_bytes_per_pixel(
                                uploaded.texture.pixel_format(),
                            )
                            .unwrap_or(uploaded.descriptor.guest.bytes_per_pixel),
                            ..uploaded.descriptor.guest
                        };
                        let native_source = match uploaded.texture.texture_type() {
                            MTLTextureType::D3 => read_texture_volume_linear(
                                &device,
                                &queue,
                                &uploaded.texture,
                                readback_descriptor,
                            ),
                            MTLTextureType::D2Array => read_texture_array_linear(
                                &device,
                                &queue,
                                &uploaded.texture,
                                readback_descriptor,
                            ),
                            _ => read_texture_linear(
                                &device,
                                &queue,
                                &uploaded.texture,
                                readback_descriptor,
                            ),
                        }
                        .map(|bytes| {
                            let depth_planes = (uploaded.texture.texture_type()
                                == MTLTextureType::D3)
                                .then(|| {
                                    diagnostic_texture_depth_plane_summaries(
                                        &bytes,
                                        readback_descriptor,
                                        uploaded.texture.depth(),
                                    )
                                });
                            (
                                bytes.len(),
                                bytes.iter().filter(|&&value| value != 0).count(),
                                diagnostic_texture_magnitude_nonzero_texels(
                                    &bytes,
                                    readback_descriptor,
                                ),
                                guest_frame_fingerprint(&bytes),
                                bytes.get(..bytes.len().min(32)).unwrap_or(&[]).to_vec(),
                                depth_planes,
                            )
                        });
                        let guest_source = snapshot_guest_texture_raw(uploaded.descriptor.guest)
                            .map(|bytes| {
                                (
                                    bytes.len(),
                                    bytes.iter().filter(|&&value| value != 0).count(),
                                    diagnostic_texture_magnitude_nonzero_texels(
                                        &bytes,
                                        uploaded.descriptor.guest,
                                    ),
                                    guest_frame_fingerprint(&bytes),
                                    bytes.get(..bytes.len().min(32)).unwrap_or(&[]).to_vec(),
                                )
                            });
                        (
                            slot,
                            binding.pc,
                            binding.opcode,
                            binding.resource_descriptor.clone(),
                            binding.sampler_descriptor.clone(),
                            uploaded.descriptor,
                            (
                                uploaded.texture.texture_type(),
                                uploaded.texture.width(),
                                uploaded.texture.height(),
                                uploaded.texture.depth(),
                                uploaded.texture.array_length(),
                                uploaded.texture.pixel_format(),
                            ),
                            native_source,
                            guest_source,
                        )
                    })
                    .collect::<Vec<_>>();
                // Do not use {:x?} on GuestTextureDescriptor: Rust's hex Debug prints
                // width/height/pitch as hex without a 0x prefix (1920→780, 1080→438), which
                // previously falsified a "half-res tex0 SRD" title-BG blocker.
                let texture_dump = texture_summaries
                    .iter()
                    .map(|summary| {
                        let guest = summary.5.guest;
                        format!(
                            "(slot={}, pc={:#x}, op={}, srd={:08x?}, samp={:08x?}, guest={{addr={:#x} {}x{} pitch={} tile={} fmt={:?} bpp={}}}, metal=({:?},{}x{}x{},arr={},{:?}), native={:x?}, guest_occ={:x?})",
                            summary.0,
                            summary.1,
                            summary.2,
                            summary.3,
                            summary.4,
                            guest.address,
                            guest.width,
                            guest.height,
                            guest.pitch,
                            guest.tile_mode,
                            guest.pixel_format,
                            guest.bytes_per_pixel,
                            summary.6.0,
                            summary.6.1,
                            summary.6.2,
                            summary.6.3,
                            summary.6.4,
                            summary.6.5,
                            summary.7,
                            summary.8,
                        )
                    })
                    .collect::<Vec<_>>();
                eprintln!(
                    "[REMu][METAL][TARGET-DRAW-TEXTURES] draw={} tex0_3={texture_dump:?}",
                    draw.sequence,
                );
                // GBuffer attribution: lighting samples large textures that were never COLOR
                // write targets (BasePass absent). Bound the first hits so title-BG runs stay
                // readable while still proving "no writer" vs "writer left zeros".
                {
                    static UNWRITTEN: AtomicUsize = AtomicUsize::new(0);
                    for summary in &texture_summaries {
                        let guest = summary.5.guest;
                        if guest.width < 64 || guest.height < 64 {
                            continue;
                        }
                        let metal_nz = summary.7.as_ref().ok().map_or(0, |v| v.1);
                        let guest_nz = summary.8.as_ref().ok().map_or(0, |v| v.1);
                        if metal_nz != 0 || guest_nz != 0 {
                            continue;
                        }
                        if diagnostic_texture_writer(guest).is_some() {
                            continue;
                        }
                        let n = UNWRITTEN.fetch_add(1, Ordering::Relaxed);
                        if n < 32 {
                            eprintln!(
                                "[REMu][METAL][GBUFFER-UNWRITTEN] draw={} sample_slot={} address={:#x} {}x{} format={:?} metal_nz=0 guest_nz=0 no_color_writer (occurrence={})",
                                draw.sequence,
                                summary.0,
                                guest.address,
                                guest.width,
                                guest.height,
                                guest.pixel_format,
                                n.saturating_add(1),
                            );
                        }
                    }
                }
                let bytes = match texture.texture_type() {
                    MTLTextureType::D3 => {
                        read_texture_volume_linear(&device, &queue, &texture, descriptor)?
                    }
                    MTLTextureType::D2Array => {
                        read_texture_array_linear(&device, &queue, &texture, descriptor)?
                    }
                    _ => read_texture_linear(&device, &queue, &texture, descriptor)?,
                };
                let nonzero = bytes.iter().filter(|&&value| value != 0).count();
                let magnitude_nonzero =
                    diagnostic_texture_magnitude_nonzero_texels(&bytes, descriptor);
                let fingerprint = guest_frame_fingerprint(&bytes);
                if diagnostic_dump_glyph_draw()
                    && evaluation_samples_glyph_atlas(pixel_evaluation)
                    && let Some(directory) = self.capture_directory.as_deref()
                {
                    match native_texture_capture_to_bgra(&bytes, descriptor) {
                        Some(bgra) => {
                            match capture_guest_frame(directory, &bgra, descriptor, fingerprint) {
                                Ok(path) => eprintln!(
                                    "[REMu][METAL][GLYPH-POST-TARGET] draw={} fingerprint={fingerprint:016x} captured={}",
                                    draw.sequence,
                                    path.display(),
                                ),
                                Err(error) => eprintln!(
                                    "[REMu][METAL][GLYPH-POST-TARGET] draw={} capture failed: {error}",
                                    draw.sequence,
                                ),
                            }
                        }
                        None => eprintln!(
                            "[REMu][METAL][GLYPH-POST-TARGET] draw={} conversion failed for {:?}",
                            draw.sequence, descriptor.pixel_format,
                        ),
                    }
                }
                let sampled_summaries = pixel_evaluation
                    .image_bindings
                    .iter()
                    .zip(&pixel_textures)
                    .filter(|(binding, _)| {
                        binding.opcode.starts_with("ImageSample")
                            || binding.opcode.starts_with("ImageGather")
                    })
                    .filter_map(|(binding, uploaded)| {
                        // A zero-copy component-reordered alias deliberately keeps its parent's
                        // Metal pixel format and applies R/B swizzles in the view. Read it back
                        // using that actual format; the guest sampling descriptor can differ.
                        let readback_descriptor = GuestTextureDescriptor {
                            pixel_format: uploaded.texture.pixel_format(),
                            bytes_per_pixel: metal_uncompressed_bytes_per_pixel(
                                uploaded.texture.pixel_format(),
                            )?,
                            ..uploaded.descriptor.guest
                        };
                        let sampled_bytes = match match uploaded.texture.texture_type() {
                            MTLTextureType::D3 => read_texture_volume_linear(
                                &device,
                                &queue,
                                &uploaded.texture,
                                readback_descriptor,
                            ),
                            MTLTextureType::D2Array => read_texture_array_linear(
                                &device,
                                &queue,
                                &uploaded.texture,
                                readback_descriptor,
                            ),
                            _ => read_texture_linear(
                                &device,
                                &queue,
                                &uploaded.texture,
                                readback_descriptor,
                            ),
                        } {
                            Ok(bytes) => bytes,
                            Err(error) => {
                                eprintln!(
                                    "[REMu][METAL][TARGET-SAMPLE-READ] pc={:#x} opcode={} guest={:?} metal={}x{} format={:?} type={:?} error={error}",
                                    binding.pc,
                                    binding.opcode,
                                    uploaded.descriptor.guest,
                                    uploaded.texture.width(),
                                    uploaded.texture.height(),
                                    uploaded.texture.pixel_format(),
                                    uploaded.texture.texture_type(),
                                );
                                return None;
                            }
                        };
                        let sampled_fingerprint = guest_frame_fingerprint(&sampled_bytes);
                        let sampled_depth_planes =
                            (uploaded.texture.texture_type() == MTLTextureType::D3).then(|| {
                                diagnostic_texture_depth_plane_summaries(
                                    &sampled_bytes,
                                    readback_descriptor,
                                    uploaded.texture.depth(),
                                )
                            });
                        if capture_target_sample_textures()
                            && let Some(directory) = self.capture_directory.as_deref()
                        {
                            let capture_descriptor = if matches!(
                                uploaded.texture.texture_type(),
                                MTLTextureType::D3 | MTLTextureType::D2Array
                            ) {
                                let slices = if uploaded.texture.texture_type()
                                    == MTLTextureType::D3
                                {
                                    uploaded.texture.depth()
                                } else {
                                    uploaded.texture.array_length()
                                };
                                let Some(height) = readback_descriptor
                                    .height
                                    .checked_mul(u32::try_from(slices).unwrap_or(u32::MAX))
                                else {
                                    eprintln!(
                                        "[REMu][METAL] could not flatten layered target sample pc={:#x}: extent overflow",
                                        binding.pc
                                    );
                                    return None;
                                };
                                GuestTextureDescriptor {
                                    height,
                                    pitch: readback_descriptor.width,
                                    tile_mode: 0,
                                    ..readback_descriptor
                                }
                            } else {
                                readback_descriptor
                            };
                            match native_texture_capture_to_bgra(
                                &sampled_bytes,
                                capture_descriptor,
                            ) {
                                Some(bgra) => match capture_guest_frame(
                                    directory,
                                    &bgra,
                                    capture_descriptor,
                                    sampled_fingerprint,
                                ) {
                                    Ok(path) => eprintln!(
                                        "[REMu][METAL] captured target sample pc={:#x} {}",
                                        binding.pc,
                                        path.display()
                                    ),
                                    Err(error) => eprintln!(
                                        "[REMu][METAL] could not capture target sample pc={:#x}: {error}",
                                        binding.pc
                                    ),
                                },
                                None => eprintln!(
                                    "[REMu][METAL] could not convert target sample pc={:#x} format={:?} bytes={}",
                                    binding.pc,
                                    readback_descriptor.pixel_format,
                                    sampled_bytes.len(),
                                ),
                            }
                        }
                        let resident = self
                            .resident_guest_textures
                            .get(&uploaded.descriptor.guest)
                            .map(|resident| (resident.gpu_written(), resident.content_version));
                        let aliases = self
                            .resident_guest_textures
                            .values()
                            .filter(|resident| {
                                guest_texture_descriptors_overlap(
                                    resident.descriptor,
                                    uploaded.descriptor.guest,
                                )
                            })
                            .map(|resident| {
                                let contents = read_texture_linear(
                                    &device,
                                    &queue,
                                    &resident.texture,
                                    resident.descriptor,
                                )
                                .ok();
                                (
                                    resident.descriptor,
                                    resident.gpu_written(),
                                    resident.content_version,
                                    contents.as_ref().map(|contents| {
                                        contents.iter().filter(|&&value| value != 0).count()
                                    }),
                                    contents
                                        .as_ref()
                                        .map(|contents| guest_frame_fingerprint(contents)),
                                    contents.as_ref().map(|contents| {
                                        (
                                            contents.iter().copied().min(),
                                            contents.iter().copied().max(),
                                            contents
                                                .get(..contents.len().min(32))
                                                .unwrap_or(&[])
                                                .to_vec(),
                                        )
                                    }),
                                )
                            })
                            .collect::<Vec<_>>();
                        let writer = diagnostic_texture_writer(uploaded.descriptor.guest);
                        let guest_raw = snapshot_guest_texture_raw(uploaded.descriptor.guest)
                            .ok()
                            .map(|raw| {
                                (
                                    raw.iter().filter(|&&value| value != 0).count(),
                                    guest_frame_fingerprint(&raw),
                                    raw.get(..raw.len().min(32)).unwrap_or(&[]).to_vec(),
                                )
                            });
                        let upstream = writer
                            .as_ref()
                            .map(|writer| {
                                writer
                                    .3
                                    .iter()
                                    .map(|descriptor| {
                                        let mut chain = Vec::new();
                                        let mut current = *descriptor;
                                        for _ in 0..12 {
                                            let resident = self
                                                .resident_guest_textures
                                                .get(&current)
                                                .map(|resident| {
                                                    let contents = read_texture_linear(
                                                        &device,
                                                        &queue,
                                                        &resident.texture,
                                                        current,
                                                    )
                                                    .ok();
                                                    (
                                                        resident.gpu_written(),
                                                        resident.content_version,
                                                        contents.as_ref().map(|contents| {
                                                            contents
                                                                .iter()
                                                                .filter(|&&value| value != 0)
                                                                .count()
                                                        }),
                                                        contents.as_ref().map(|contents| {
                                                            guest_frame_fingerprint(contents)
                                                        }),
                                                    )
                                                });
                                            let cached = self
                                                .sampled_texture_cache
                                                .iter()
                                                .filter(|(key, _)| key.descriptor == current)
                                                .max_by_key(|(_, cached)| cached.last_use)
                                                .and_then(|(_, cached)| {
                                                    let contents = read_texture_linear(
                                                        &device,
                                                        &queue,
                                                        &cached.texture,
                                                        current,
                                                    )
                                                    .ok()?;
                                                    Some((
                                                        contents
                                                            .iter()
                                                            .filter(|&&value| value != 0)
                                                            .count(),
                                                        guest_frame_fingerprint(&contents),
                                                    ))
                                                });
                                            let aliases = self
                                                .resident_guest_textures
                                                .values()
                                                .filter(|resident| {
                                                    guest_texture_descriptors_overlap(
                                                        resident.descriptor,
                                                        current,
                                                    )
                                                })
                                                .map(|resident| {
                                                    (
                                                        resident.descriptor,
                                                        resident.gpu_written(),
                                                        resident.content_version,
                                                    )
                                                })
                                                .collect::<Vec<_>>();
                                            let current_writer = diagnostic_texture_writer(current);
                                            let overlapping_writers =
                                                diagnostic_overlapping_texture_writers(current);
                                            let memory_copies =
                                                diagnostic_overlapping_memory_copies(current);
                                            chain.push((
                                                current,
                                                resident,
                                                cached,
                                                aliases,
                                                current_writer.clone(),
                                                overlapping_writers,
                                                memory_copies,
                                            ));
                                            let Some(next) = current_writer
                                                .as_ref()
                                                .and_then(|writer| writer.3.first())
                                                .copied()
                                            else {
                                                break;
                                            };
                                            if chain.iter().any(|(descriptor, _, _, _, _, _, _)| {
                                                *descriptor == next
                                            }) {
                                                break;
                                            }
                                            current = next;
                                        }
                                        chain
                                    })
                                    .collect::<Vec<_>>()
                            })
                            .unwrap_or_default();
                        Some((
                            (
                                binding.pc,
                                binding.opcode,
                                binding.sampler_descriptor.clone(),
                                uploaded.descriptor.texture_type,
                                uploaded.descriptor.layer_count,
                                uploaded.texture.texture_type(),
                                uploaded.texture.array_length(),
                            ),
                            uploaded.descriptor.guest,
                            resident,
                            writer,
                            diagnostic_overlapping_texture_writers(uploaded.descriptor.guest),
                            diagnostic_related_texture_writers(uploaded.descriptor.guest),
                            (
                                diagnostic_overlapping_memory_copies(uploaded.descriptor.guest),
                                diagnostic_overlapping_global_writers(uploaded.descriptor.guest),
                            ),
                            (guest_raw, upstream),
                            aliases,
                            (
                                sampled_bytes.iter().filter(|&&value| value != 0).count(),
                                sampled_fingerprint,
                                sampled_depth_planes,
                                diagnostic_texture_numeric_range(
                                    &sampled_bytes,
                                    readback_descriptor,
                                ),
                                sampled_bytes
                                    .get(..sampled_bytes.len().min(32))
                                    .unwrap_or(&[])
                                    .to_vec(),
                            ),
                        ))
                    })
                    .collect::<Vec<_>>();
                // Same {:x?} footgun as TARGET-DRAW-TEXTURES: print guest extents in decimal.
                let sampled_dump = sampled_summaries
                    .iter()
                    .map(|summary| {
                        let guest = summary.1;
                        format!(
                            "(meta={:x?}, guest={{addr={:#x} {}x{} pitch={} tile={} fmt={:?} bpp={}}}, resident={:x?}, writer={:x?}, overlap_writers={:x?}, related_writers={:x?}, mem={:x?}, raw_upstream={:x?}, aliases={:x?}, occ={:x?})",
                            summary.0,
                            guest.address,
                            guest.width,
                            guest.height,
                            guest.pitch,
                            guest.tile_mode,
                            guest.pixel_format,
                            guest.bytes_per_pixel,
                            summary.2,
                            summary.3,
                            summary.4,
                            summary.5,
                            summary.6,
                            summary.7,
                            summary.8,
                            summary.9,
                        )
                    })
                    .collect::<Vec<_>>();
                let rgba8_summary = if matches!(
                    descriptor.pixel_format,
                    MTLPixelFormat::RGBA8Unorm | MTLPixelFormat::BGRA8Unorm
                ) {
                    // BGRA and RGBA share the same 8-bit layout for occupancy: bytes 0..2 are the
                    // color channels in Metal getBytes order (B,G,R or R,G,B). Count RGB-nonzero
                    // pixels so a clear-alpha-only surface cannot look "occupied".
                    let first = bytes.get(..4).unwrap_or(&[]);
                    let mut minimum = [u8::MAX; 4];
                    let mut maximum = [u8::MIN; 4];
                    let mut different = 0_usize;
                    let mut rgb_nonzero = 0_usize;
                    for pixel in bytes.chunks_exact(4) {
                        if pixel != first {
                            different += 1;
                        }
                        if pixel[0] != 0 || pixel[1] != 0 || pixel[2] != 0 {
                            rgb_nonzero += 1;
                        }
                        for component in 0..4 {
                            minimum[component] = minimum[component].min(pixel[component]);
                            maximum[component] = maximum[component].max(pixel[component]);
                        }
                    }
                    format!(
                        " first={first:02x?} min={minimum:02x?} max={maximum:02x?} different={different} rgb_nonzero={rgb_nonzero}"
                    )
                } else if descriptor.pixel_format == MTLPixelFormat::RGBA16Float {
                    const MAGENTA_RGBA16F: [u8; 8] = [0x00, 0x3c, 0, 0, 0x00, 0x3c, 0x00, 0x3c];
                    const WHITE_RGBA16F: [u8; 8] = [0x00, 0x3c, 0x00, 0x3c, 0x00, 0x3c, 0x00, 0x3c];
                    let first = bytes.get(..8).unwrap_or(&[]);
                    let magenta = bytes
                        .chunks_exact(8)
                        .filter(|pixel| *pixel == MAGENTA_RGBA16F)
                        .count();
                    let white = bytes
                        .chunks_exact(8)
                        .filter(|pixel| *pixel == WHITE_RGBA16F)
                        .count();
                    let zero = bytes
                        .chunks_exact(8)
                        .filter(|pixel| pixel.iter().all(|&component| component == 0))
                        .count();
                    let different = bytes
                        .chunks_exact(8)
                        .filter(|pixel| *pixel != first)
                        .count();
                    format!(
                        " first={first:02x?} magenta={magenta} white={white} zero={zero} different={different}"
                    )
                } else {
                    String::new()
                };
                let numeric_range = diagnostic_texture_numeric_range(&bytes, descriptor);
                let diagnostic_vertex_positions = self
                    .diagnostic_vertex_position_buffer
                    .as_ref()
                    .and_then(|buffer| {
                        let word_count = DIAGNOSTIC_VERTEX_POSITION_VERTEX_CAPACITY
                            .checked_mul(DIAGNOSTIC_VERTEX_STATE_WORDS_PER_VERTEX)?;
                        // The diagnostic buffer is allocated above with exactly this many shared
                        // u32 words and the completed command batch is waited before readback.
                        let words =
                            super::MetalBytes::new(buffer).typed_slice::<u32>(0, word_count)?;
                        let indices = if uses_mesh_stage {
                            // The injected mesh capture writes one record per Metal mesh-local
                            // thread. Guest indices are data read by those threads and do not name
                            // records in this buffer.
                            (0..64).collect::<Vec<_>>()
                        } else {
                            let mut seen = HashSet::new();
                            indexed_draw
                                .as_ref()?
                                .0
                                .iter()
                                .copied()
                                .filter_map(|index| {
                                    let adjusted = i64::from(index) + i64::from(draw.base_vertex);
                                    u32::try_from(adjusted).ok()
                                })
                                .filter(|index| {
                                    usize::try_from(*index).is_ok_and(|index| {
                                        index < DIAGNOSTIC_VERTEX_POSITION_VERTEX_CAPACITY
                                    }) && seen.insert(*index)
                                })
                                .take(24)
                                .collect()
                        };
                        Some(
                            indices
                                .into_iter()
                                .map(|index| {
                                    let base = usize::try_from(index)
                                        .expect("filtered diagnostic vertex index fits usize")
                                        * DIAGNOSTIC_VERTEX_STATE_WORDS_PER_VERTEX;
                                    let raw = [
                                        words[base],
                                        words[base + 1],
                                        words[base + 2],
                                        words[base + 3],
                                    ];
                                    let value = raw.map(f32::from_bits);
                                    let registers = words
                                        [base + 4..base + DIAGNOSTIC_VERTEX_STATE_WORDS_PER_VERTEX]
                                        .to_vec();
                                    (index, raw, value, registers)
                                })
                                .collect::<Vec<_>>(),
                        )
                    })
                    .unwrap_or_default();
                let diagnostic_pixel_outputs = self
                    .diagnostic_pixel_output_buffer
                    .as_ref()
                    .and_then(|buffer| {
                        let word_count = DIAGNOSTIC_PIXEL_OUTPUT_WIDTH
                            .checked_mul(DIAGNOSTIC_PIXEL_OUTPUT_HEIGHT)?
                            .checked_mul(DIAGNOSTIC_PIXEL_OUTPUT_WORDS_PER_PIXEL)?;
                        // The shared diagnostic buffer is allocated with exactly this many words
                        // and the command batch has completed before this readback.
                        let words =
                            super::MetalBytes::new(buffer).typed_slice::<u32>(0, word_count)?;
                        let covered = words
                            .chunks_exact(DIAGNOSTIC_PIXEL_OUTPUT_WORDS_PER_PIXEL)
                            .filter(|pixel| pixel[0] != 0)
                            .count();
                        // Count magnitude, not raw bits: IEEE -0.0 is 0x80000000 and must not
                        // look like a live HDR export (SH title 800bf4).
                        let nonzero = words
                            .chunks_exact(DIAGNOSTIC_PIXEL_OUTPUT_WORDS_PER_PIXEL)
                            .filter(|pixel| {
                                pixel[0] != 0
                                    && pixel[3..]
                                        .iter()
                                        .copied()
                                        .any(diagnostic_f32_bits_magnitude_nonzero)
                            })
                            .count();
                        let mut finite = 0_usize;
                        let mut sum = [0.0_f64; 4];
                        let mut minimum = [f32::INFINITY; 4];
                        let mut maximum = [f32::NEG_INFINITY; 4];
                        let mut above_one = [0_usize; 4];
                        let mut uv_sum = [0.0_f64; 2];
                        let mut uv_minimum = [f32::INFINITY; 2];
                        let mut uv_maximum = [f32::NEG_INFINITY; 2];
                        let mut texture_sample_sum = [0.0_f64; 4];
                        let mut texture_sample_minimum = [f32::INFINITY; 4];
                        let mut texture_sample_maximum = [f32::NEG_INFINITY; 4];
                        for pixel in words
                            .chunks_exact(DIAGNOSTIC_PIXEL_OUTPUT_WORDS_PER_PIXEL)
                            .filter(|pixel| pixel[0] != 0)
                        {
                            for component in 0..2 {
                                let value = f32::from_bits(pixel[component + 1]);
                                if value.is_finite() {
                                    uv_sum[component] += f64::from(value);
                                    uv_minimum[component] = uv_minimum[component].min(value);
                                    uv_maximum[component] = uv_maximum[component].max(value);
                                }
                            }
                            let values: [f32; 4] = std::array::from_fn(|component| {
                                f32::from_bits(pixel[component + 3])
                            });
                            if values.iter().all(|value| value.is_finite()) {
                                finite += 1;
                                for component in 0..4 {
                                    sum[component] += f64::from(values[component]);
                                    minimum[component] = minimum[component].min(values[component]);
                                    maximum[component] = maximum[component].max(values[component]);
                                    above_one[component] += usize::from(values[component] > 1.0);
                                }
                            }
                            let texture_sample: [f32; 4] = std::array::from_fn(|component| {
                                f32::from_bits(pixel[component + 7])
                            });
                            if texture_sample.iter().all(|value| value.is_finite()) {
                                for component in 0..4 {
                                    texture_sample_sum[component] +=
                                        f64::from(texture_sample[component]);
                                    texture_sample_minimum[component] = texture_sample_minimum
                                        [component]
                                        .min(texture_sample[component]);
                                    texture_sample_maximum[component] = texture_sample_maximum
                                        [component]
                                        .max(texture_sample[component]);
                                }
                            }
                        }
                        let mean = sum.map(|value| {
                            if finite == 0 {
                                0.0
                            } else {
                                value / finite as f64
                            }
                        });
                        let uv_mean = uv_sum.map(|value| {
                            if covered == 0 {
                                0.0
                            } else {
                                value / covered as f64
                            }
                        });
                        let texture_sample_mean = texture_sample_sum.map(|value| {
                            if covered == 0 {
                                0.0
                            } else {
                                value / covered as f64
                            }
                        });
                        let first = words
                            .chunks_exact(DIAGNOSTIC_PIXEL_OUTPUT_WORDS_PER_PIXEL)
                            .enumerate()
                            .filter(|(_, pixel)| pixel[0] != 0)
                            .take(8)
                            .map(|(pixel, words)| {
                                let uv_raw = [words[1], words[2]];
                                let uv = uv_raw.map(f32::from_bits);
                                let raw = [words[3], words[4], words[5], words[6]];
                                let value = raw.map(f32::from_bits);
                                let texture_sample_raw =
                                    [words[7], words[8], words[9], words[10]];
                                let texture_sample = texture_sample_raw.map(f32::from_bits);
                                (
                                    pixel % DIAGNOSTIC_PIXEL_OUTPUT_WIDTH,
                                    pixel / DIAGNOSTIC_PIXEL_OUTPUT_WIDTH,
                                    uv_raw,
                                    uv,
                                    raw,
                                    value,
                                    texture_sample_raw,
                                    texture_sample,
                                )
                            })
                            .collect::<Vec<_>>();
                        Some((
                            covered,
                            nonzero,
                            finite,
                            uv_minimum,
                            uv_maximum,
                            uv_mean,
                            minimum,
                            maximum,
                            mean,
                            above_one,
                            (
                                texture_sample_minimum,
                                texture_sample_maximum,
                                texture_sample_mean,
                            ),
                            first,
                        ))
                    });
                if diagnostic_gbuffer_draw
                    && let Some((binding, uploaded)) = pixel_evaluation
                        .global_memory_bindings
                        .iter()
                        .zip(&pixel_buffers)
                        .find(|(binding, _)| binding.scalar_address == 8)
                {
                    let start = usize::try_from(uploaded.offset)
                        .map_err(|_| "diagnostic G-buffer global offset does not fit usize")?;
                    let bytes = super::MetalBytes::new(&uploaded.buffer)
                        .slice(start, binding.byte_count)
                        .ok_or("diagnostic G-buffer global is not CPU-visible")?;
                    let dword = |offset: usize| {
                        bytes
                            .get(offset..offset.saturating_add(4))
                            .map(|word| u32::from_le_bytes([word[0], word[1], word[2], word[3]]))
                    };
                    let records = (0_usize..256)
                        .filter_map(|primitive| {
                            let base = primitive.checked_mul(36)?.checked_mul(16)?;
                            let field20 = dword(base.checked_add(20 * 16 + 12)?)?;
                            let field26 = dword(base.checked_add(26 * 16 + 12)?)?;
                            (field20 != 0 || field26 != 0).then_some((primitive, field20, field26))
                        })
                        .collect::<Vec<_>>();
                    eprintln!(
                        "[REMu][METAL][GBUFFER-GLOBAL] draw={} address={:#x} bytes={:#x} nonzero_records={records:08x?}",
                        draw.sequence, binding.base_address, binding.byte_count,
                    );
                }
                eprintln!(
                    "[REMu][METAL][TARGET] draw={} es={:#x} ps={:#x} indexed={} count={} instances={} primitive={:#x} index_offset={} base_vertex={} indices={:?} address={:#x} {}x{} format={:?} bytes={} nonzero={} magnitude_nz={} numeric={numeric_range:?} fingerprint={fingerprint:016x}{rgba8_summary} pipeline_targets={:?} vs_out={:#010x} ps_inputs={:08x?} vertex_positions={diagnostic_vertex_positions:?} pixel_outputs={diagnostic_pixel_outputs:?} sampled={sampled_dump:?} vertex_scalars={:08x?} pixel_scalars={:08x?} pixel_final_scalars={:08x?} pixel_scalar_memory_reads={:x?} vertex_globals_evaluated={:x?} pixel_globals_evaluated={:x?} vertex_globals_submitted={:x?} pixel_globals_submitted={:x?}",
                    draw.sequence,
                    draw.export_shader_address,
                    draw.pixel_shader_address,
                    draw.indexed,
                    draw.index_count,
                    draw.instance_count,
                    draw.primitive_type,
                    draw.index_offset,
                    draw.base_vertex,
                    indexed_draw
                        .as_ref()
                        .map(|(indices, _)| &indices[..indices.len().min(24)]),
                    descriptor.address,
                    descriptor.width,
                    descriptor.height,
                    descriptor.pixel_format,
                    bytes.len(),
                    nonzero,
                    magnitude_nonzero,
                    key.targets,
                    key.vertex_output_control,
                    key.pixel_input_controls,
                    vertex_scalar_metadata,
                    pixel_evaluation.initial_scalar_registers,
                    pixel_evaluation.scalar_registers,
                    pixel_evaluation.scalar_memory_reads,
                    export_evaluation
                        .global_memory_bindings
                        .iter()
                        .map(|binding| (
                            binding.scalar_address,
                            binding.base_address,
                            binding.byte_count,
                            diagnostic_global_binding_head(binding),
                        ))
                        .collect::<Vec<_>>(),
                    pixel_evaluation
                        .global_memory_bindings
                        .iter()
                        .map(|binding| (
                            binding.scalar_address,
                            binding.base_address,
                            binding.byte_count,
                            diagnostic_global_binding_head(binding),
                        ))
                        .collect::<Vec<_>>(),
                    export_evaluation
                        .global_memory_bindings
                        .iter()
                        .zip(&export_buffers)
                        .map(|(binding, uploaded)| (
                            binding.scalar_address,
                            binding.base_address,
                            binding.byte_count,
                            binding.writable,
                            diagnostic_uploaded_global_binding_head(binding, uploaded),
                        ))
                        .collect::<Vec<_>>(),
                    pixel_evaluation
                        .global_memory_bindings
                        .iter()
                        .zip(&pixel_buffers)
                        .map(|(binding, uploaded)| (
                            binding.scalar_address,
                            binding.base_address,
                            binding.byte_count,
                            binding.writable,
                            diagnostic_uploaded_global_binding_head(binding, uploaded),
                        ))
                        .collect::<Vec<_>>(),
                );
                // A requested one-shot target dump must not be suppressed by the periodic
                // flip-capture interval. It is commonly used to inspect two adjacent draws in
                // the same frame, where the normal capture throttle intentionally keeps only one.
                self.last_guest_capture = None;
                self.capture_translated_frame_if_changed(descriptor, &bytes);
            }
        }
        if diagnostic_trace_target_contents()
            && draw.sequence >= diagnostic_dump_target_minimum_draw()
            && draw.sequence <= diagnostic_trace_target_maximum_draw()
            && (draw.sequence - diagnostic_dump_target_minimum_draw())
                .is_multiple_of(diagnostic_trace_target_stride())
            && let Some(target) = targets.iter().find(|target| {
                target.sample_count > 1
                    && target.texture.width == 1_920
                    && target.texture.height == 1_080
                    && target.texture.pixel_format == MTLPixelFormat::RGBA16Float
            })
        {
            self.wait_for_guest_command_batches()?;
            let descriptor = target.texture;
            let texture = self
                .resident_guest_textures
                .get(&descriptor)
                .filter(|resident| resident.serves_reads())
                .ok_or("traced target has no resident Metal texture")?
                .texture
                .clone();
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
            let bytes = read_texture_linear(&device, &queue, &texture, descriptor)?;
            const WHITE_RGBA16F: [u8; 8] = [0x00, 0x3c, 0x00, 0x3c, 0x00, 0x3c, 0x00, 0x3c];
            let pixels = bytes.chunks_exact(8);
            let pixel_count = pixels.len();
            let zero = pixels
                .clone()
                .filter(|pixel| pixel.iter().all(|&component| component == 0))
                .count();
            let white = pixels
                .clone()
                .filter(|pixel| *pixel == WHITE_RGBA16F)
                .count();
            let mut sums = [0.0_f64; 4];
            let mut minimum = [f32::INFINITY; 4];
            let mut maximum = [f32::NEG_INFINITY; 4];
            let mut finite_pixels = 0_usize;
            for pixel in pixels.clone() {
                let values: [f32; 4] = std::array::from_fn(|component| {
                    let offset = component * 2;
                    half_float_to_f32(u16::from_le_bytes([pixel[offset], pixel[offset + 1]]))
                });
                if values.iter().all(|value| value.is_finite()) {
                    finite_pixels += 1;
                    for component in 0..4 {
                        sums[component] += f64::from(values[component]);
                        minimum[component] = minimum[component].min(values[component]);
                        maximum[component] = maximum[component].max(values[component]);
                    }
                }
            }
            let mean = sums.map(|sum| {
                if finite_pixels == 0 {
                    0.0
                } else {
                    sum / finite_pixels as f64
                }
            });
            let class = if zero == pixel_count {
                0_u8
            } else if white == pixel_count {
                2_u8
            } else {
                1_u8
            };
            static LAST_CLASSES: OnceLock<Mutex<HashMap<u64, u8>>> = OnceLock::new();
            let changed = LAST_CLASSES
                .get_or_init(|| Mutex::new(HashMap::new()))
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner)
                .insert(descriptor.address, class)
                != Some(class);
            if changed || diagnostic_trace_all_target_contents() {
                let class_name = match class {
                    0 => "zero",
                    2 => "white",
                    _ => "mixed",
                };
                eprintln!(
                    "[REMu][METAL][TARGET-CONTENT] draw={} ps={:#x} ps_insns={} count={} instances={} address={:#x} class={} zero={} white={} pixels={} finite={} min={minimum:?} max={maximum:?} mean={mean:?} fingerprint={:016x} mask={:#x} aa_mask={:08x}/{:08x} scissor={scissor:?} viewport=({:.1},{:.1} {:.1}x{:.1} {:.3}..{:.3}) depth={depth:?} raster={raster:?}",
                    draw.sequence,
                    draw.pixel_shader_address,
                    draw.pixel_shader
                        .as_ref()
                        .map_or(0, |shader| shader.program.instructions.len()),
                    draw.index_count,
                    draw.instance_count,
                    descriptor.address,
                    class_name,
                    zero,
                    white,
                    pixel_count,
                    finite_pixels,
                    guest_frame_fingerprint(&bytes),
                    target.write_mask,
                    register_value(&draw.context_registers, PA_SC_AA_MASK_X0Y0_X1Y0)
                        .unwrap_or(u32::MAX),
                    register_value(&draw.context_registers, PA_SC_AA_MASK_X0Y1_X1Y1)
                        .unwrap_or(u32::MAX),
                    viewport.originX,
                    viewport.originY,
                    viewport.width,
                    viewport.height,
                    viewport.znear,
                    viewport.zfar,
                );
            }
        }
        if let (Some(encoding_started), Some(encoding_finished)) =
            (ack_metal_encoding_started, ack_metal_encoding_finished)
        {
            let finished = Instant::now();
            self.record_shared_event_ack_batch_cpu_phases(
                encoding_started
                    .saturating_duration_since(started)
                    .saturating_add(finished.saturating_duration_since(encoding_finished)),
                encoding_finished.saturating_duration_since(encoding_started),
                false,
            );
        }
        let batch_full = self
            .active_open_guest_batch()
            .is_some_and(|batch| batch.work.len() >= GUEST_BATCH_WORK_CAPACITY);
        if batch_full {
            self.flush_guest_command_batch(
                FlushPolicy::Opportunistic,
                GuestBatchFlushReason::WorkCapacity,
            )?;
        }
        if let Some(directory) = std::env::var_os("REMU_DUMP_VIDEO_TARGET_DIR")
            && evaluation_samples_large_linear_video(pixel_evaluation)
            && {
                static VIDEO_TARGET_DUMP_MATCHES: AtomicUsize = AtomicUsize::new(0);
                VIDEO_TARGET_DUMP_MATCHES.fetch_add(1, Ordering::Relaxed)
                    == diagnostic_dump_target_match_index()
            }
        {
            self.wait_for_guest_command_batches()?;
            let descriptor = targets
                .first()
                .expect("render-target decoder requires at least one target")
                .texture;
            let texture = self
                .resident_guest_textures
                .get(&descriptor)
                .filter(|resident| resident.serves_reads())
                .ok_or("video render target has no resident Metal texture")?
                .texture
                .clone();
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
            let bytes = read_texture_linear(&device, &queue, &texture, descriptor)?;
            let directory = Path::new(&directory);
            std::fs::create_dir_all(directory)
                .map_err(|error| format!("could not create {}: {error}", directory.display()))?;
            let path = directory.join(format!(
                "draw-{}-target-{:x}-{}x{}-{:?}.bin",
                draw.sequence,
                descriptor.address,
                descriptor.width,
                descriptor.height,
                descriptor.pixel_format,
            ));
            std::fs::write(&path, &bytes)
                .map_err(|error| format!("could not write {}: {error}", path.display()))?;
            for (index, (_binding, uploaded)) in pixel_evaluation
                .image_bindings
                .iter()
                .zip(&pixel_textures)
                .enumerate()
                .filter(|(_, (binding, _))| {
                    binding.opcode.starts_with("ImageSample")
                        || binding.opcode.starts_with("ImageGather")
                })
            {
                let sampled_descriptor = uploaded.descriptor.guest;
                let sampled_bytes =
                    read_texture_linear(&device, &queue, &uploaded.texture, sampled_descriptor)?;
                let sampled_path = directory.join(format!(
                    "draw-{}-sample-{index}-{:x}-{}x{}-{:?}.bin",
                    draw.sequence,
                    sampled_descriptor.address,
                    sampled_descriptor.width,
                    sampled_descriptor.height,
                    sampled_descriptor.pixel_format,
                ));
                std::fs::write(&sampled_path, &sampled_bytes).map_err(|error| {
                    format!("could not write {}: {error}", sampled_path.display())
                })?;
                if let Ok(raw) = snapshot_guest_texture_raw(sampled_descriptor) {
                    let raw_path = directory.join(format!(
                        "draw-{}-sample-{index}-{:x}-guest-raw.bin",
                        draw.sequence, sampled_descriptor.address,
                    ));
                    std::fs::write(&raw_path, raw).map_err(|error| {
                        format!("could not write {}: {error}", raw_path.display())
                    })?;
                }
            }
            let scalar_path = directory.join(format!("draw-{}-export-scalars.txt", draw.sequence));
            std::fs::write(
                &scalar_path,
                format!("{:08x?}\n", export_evaluation.initial_scalar_registers),
            )
            .map_err(|error| format!("could not write {}: {error}", scalar_path.display()))?;
            for (index, (binding, uploaded)) in export_evaluation
                .global_memory_bindings
                .iter()
                .zip(&export_buffers)
                .enumerate()
                .filter(|(_, (binding, _))| binding.byte_count <= 4 * 1024 * 1024)
            {
                let offset = usize::try_from(uploaded.offset)
                    .map_err(|_| "video global-buffer offset does not fit usize")?;
                // Every uploaded global buffer is shared CPU-visible storage; skip any that is not
                // mapped or whose validated extent no longer fits (null contents -> zero length).
                let uploaded_bytes = super::MetalBytes::new(&uploaded.buffer);
                let Some(data) = uploaded_bytes.slice(offset, binding.byte_count) else {
                    continue;
                };
                let binding_path = directory.join(format!(
                    "draw-{}-export-binding-{index}-{:x}-{}.bin",
                    draw.sequence, binding.base_address, binding.byte_count,
                ));
                std::fs::write(&binding_path, data).map_err(|error| {
                    format!("could not write {}: {error}", binding_path.display())
                })?;
            }
            eprintln!(
                "[REMu][METAL][VIDEO] dumped immediate target {} ({} bytes)",
                path.display(),
                bytes.len()
            );
        }
        if trace {
            eprintln!(
                "[REMu][METAL] encoded Gen5 draw={} es={:#x} ps={:#x} in {:?}",
                draw.sequence,
                key.export_shader_address,
                key.pixel_shader_address,
                started.elapsed()
            );
        }
        Ok(())
    }

    /// Executes the command-processor color resolve defined by MRT0 -> MRT1.
    ///
    /// Ordinary Metal render passes already resolve REmu's private multisample attachment into a
    /// single-sample resident for MRT0. The guest resolve operation still names distinct
    /// destination storage, so that current resident must be copied to MRT1 before post-processing
    /// samples it. shadPS4's Vulkan path performs the same explicit MRT0-to-MRT1 transfer.
    fn execute_color_resolve(
        &mut self,
        draw: &IndexedDraw,
    ) -> Result<(GuestTextureDescriptor, GuestTextureDescriptor), String> {
        let (source, destination) = decode_color_resolve_targets(draw)?;
        if source.texture_type != 9
            || destination.texture_type != 9
            || source.layer_count != 1
            || destination.layer_count != 1
            || source.mip_level != 0
            || destination.mip_level != 0
            || source.mip_level_count != 1
            || destination.mip_level_count != 1
        {
            return Err(format!(
                "color resolve requires base-level 2D surfaces, got source={source:?} destination={destination:?}"
            ));
        }
        let device = self
            .device
            .as_ref()
            .ok_or("Metal device is unavailable")?
            .clone();
        let source_texture = self
            .ensure_layered_render_target_texture(&device, source, false)?
            .texture;
        let destination_resident =
            self.ensure_layered_render_target_texture(&device, destination, true)?;
        let destination_texture = destination_resident.texture.clone();
        let activity_stamp = ResidentTextureActivityStamp {
            work: draw.sequence,
            flip: self.next_flip_sequence,
        };
        self.note_resident_texture_read_at(source.texture, activity_stamp);

        self.ensure_active_open_guest_batch()?;
        let batch = self
            .active_open_guest_batch_mut()
            .expect("color resolve opened a guest command batch");
        batch.close_active_pass_encoders();
        let encoder = batch.command_buffer.new_blit_command_encoder();
        encoder.set_label("REMu translated MRT0-to-MRT1 color resolve");
        encoder.copy_from_texture(
            &source_texture,
            0,
            0,
            MTLOrigin { x: 0, y: 0, z: 0 },
            MTLSize::new(
                u64::from(source.texture.width),
                u64::from(source.texture.height),
                1,
            ),
            &destination_texture,
            0,
            0,
            MTLOrigin { x: 0, y: 0, z: 0 },
        );
        encoder.end_encoding();
        batch.work.push(GuestBatchWork::Draw {
            sequence: draw.sequence,
            indexed: draw.indexed,
            indirect_arguments_address: draw.indirect_arguments_address,
            export: draw.export_shader_address,
            pixel: draw.pixel_shader_address,
            index_count: draw.index_count,
            instance_count: draw.instance_count,
            target: draw.color_target_address,
            extent: [destination.texture.width, destination.texture.height],
            export_msl_bytes: 0,
            pixel_msl_bytes: 0,
            export_dispatcher: false,
            pixel_dispatcher: false,
        });
        self.mark_resident_texture_written_at(
            destination.texture,
            destination.span,
            Some(destination_resident.descriptor),
            // The resolve draw covers the whole base-level destination surface.
            WrittenRegion::whole_resource(),
            activity_stamp,
        );
        self.note_pending_flip_target_write(destination.texture);
        Ok((source.texture, destination.texture))
    }

    /// Materializes the logical color represented by MRT0's fast-clear metadata.
    ///
    /// shadPS4 clears here only while CMASK still marks the selected slice as cleared. REmu keeps
    /// render targets in uncompressed Metal storage, so ordinary attachment begin consumes that
    /// state through a Metal load clear. An elimination after the attachment was drawn is then a
    /// no-op instead of erasing the completed frame.
    fn execute_eliminate_fast_clear(
        &mut self,
        draw: &IndexedDraw,
    ) -> Result<Option<GuestTextureDescriptor>, String> {
        let Some(target) = decode_color_eliminate_fast_clear_target(draw)? else {
            super::stats::NOOP_DRAWS_FAST_CLEAR_NOT_PENDING.fetch_add(1, Ordering::Relaxed);
            super::stats::note_noop_draw("gpu:noop-fast-clear-idle");
            return Ok(None);
        };
        if !consume_color_metadata_clear(
            &mut self.color_metadata_clears,
            render_target_cmask_address(&target, &draw.context_registers)?,
            target.base_layer,
        ) {
            super::stats::NOOP_DRAWS_FAST_CLEAR_NOT_PENDING.fetch_add(1, Ordering::Relaxed);
            super::stats::note_noop_draw("gpu:noop-fast-clear-idle");
            return Ok(None);
        }
        let activity_stamp = ResidentTextureActivityStamp {
            work: draw.sequence,
            flip: self.next_flip_sequence,
        };
        if target.texture_type != 9
            || target.layer_count != 1
            || target.mip_level != 0
            || target.mip_level_count != 1
        {
            return Err(format!(
                "fast-clear elimination requires a base-level 2D surface, got {target:?}"
            ));
        }
        let clear_color = render_target_fast_clear_color(&target, &draw.context_registers)
            .ok_or("fast-clear elimination has no decodable clear color")?;
        let device = self
            .device
            .as_ref()
            .ok_or("Metal device is unavailable")?
            .clone();
        let selected = self.ensure_layered_render_target_texture(&device, target, true)?;
        let resolved = selected.texture.clone();
        let (attachment_texture, resolve_texture) = if target.sample_count > 1 {
            let (multisample, _) = self.ensure_multisample_texture(
                &device,
                target.texture,
                target.sample_count,
                target.texture.width,
                target.texture.height,
            )?;
            (multisample, Some(resolved))
        } else {
            (resolved, None)
        };

        self.ensure_active_open_guest_batch()?;
        let pass = RenderPassDescriptor::new();
        pass.set_render_target_width(u64::from(target.texture.width));
        pass.set_render_target_height(u64::from(target.texture.height));
        let attachment = pass
            .color_attachments()
            .object_at(0)
            .ok_or("Metal render pass has no color attachment zero")?;
        attachment.set_texture(Some(&attachment_texture));
        attachment.set_load_action(MTLLoadAction::Clear);
        attachment.set_clear_color(clear_color);
        if let Some(resolve_texture) = resolve_texture.as_ref() {
            attachment.set_resolve_texture(Some(resolve_texture));
        }
        let memoryless = super::attachment::msaa_memoryless_decision(
            self.statistics.encoded_draws.saturating_add(1),
            self.statistics.encoded_render_passes.saturating_add(1),
            false,
        );
        attachment.set_store_action(
            super::attachment::color_store_action(resolve_texture.is_some(), &memoryless).metal(),
        );

        let batch = self
            .active_open_guest_batch_mut()
            .expect("fast-clear elimination opened a guest command batch");
        batch.close_active_pass_encoders();
        let encoder = batch.command_buffer.new_render_command_encoder(pass);
        encoder.set_label("REMu translated color fast-clear elimination");
        encoder.end_encoding();
        batch.work.push(GuestBatchWork::Draw {
            sequence: draw.sequence,
            indexed: draw.indexed,
            indirect_arguments_address: draw.indirect_arguments_address,
            export: draw.export_shader_address,
            pixel: draw.pixel_shader_address,
            index_count: draw.index_count,
            instance_count: draw.instance_count,
            target: draw.color_target_address,
            extent: [target.texture.width, target.texture.height],
            export_msl_bytes: 0,
            pixel_msl_bytes: 0,
            export_dispatcher: false,
            pixel_dispatcher: false,
        });
        // Materializing fast-clear metadata rewrites every texel of the base-level target.
        let region = WrittenRegion::whole_resource();
        if target.sample_count > 1 {
            self.mark_multisample_texture_initialized(target.texture, target.sample_count);
            self.mark_resident_texture_written_preserving_multisample_at(
                target.texture,
                target.span,
                Some(selected.descriptor),
                region,
                activity_stamp,
            );
        } else {
            self.mark_resident_texture_written_at(
                target.texture,
                target.span,
                Some(selected.descriptor),
                region,
                activity_stamp,
            );
        }
        self.note_pending_flip_target_write(target.texture);
        Ok(Some(target.texture))
    }
}
