// Copyright (C) 2026 REmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

use super::super::cache::metal_buffer_with_guest_backing_range;
use super::super::completion::commit_and_settle;
use super::super::tiler::{
    encode_gpu_upload_from_guest, gpu_tiler_owns_descriptor, mode27_tiler_pipelines,
};
use super::super::{
    FlushPolicy, GuestBatchFlushReason, SharedEventResidentTextureVersion, SharedEventWriteKind,
    WritebackTrigger,
};
use super::{
    DecodedDepthTarget, DecodedImageDescriptor, DecodedRenderTarget, DirectScanoutTexture,
    GuestTextureDescriptor, MetalPresenter, Mode27WorkOrigin, MultisampleResidentTexture,
    MultisampleTextureKey, OWNED_REMATERIALISE_BLIT, OWNED_REMATERIALISE_CPU,
    OWNED_REMATERIALISE_DETILE, OWNED_REMATERIALISE_FAILED, OWNED_REMATERIALISE_FAILED_CPU_GPU,
    OWNED_REMATERIALISE_FAILED_LAYOUT, OWNED_REMATERIALISE_FAILED_MAPPING,
    OWNED_REMATERIALISE_FAILED_METAL, OWNED_REMATERIALISE_VIEW, OWNED_SUBTRACT_TENANTS_SURVIVED,
    RESIDENT_WRITE_ATTRIBUTION_MISSES, RangeSet, ResidencyState, ResidentGuestTexture,
    ResidentTextureActivityStamp, ResidentTextureMap, ResourceSpan, ScanoutSnapshotTexture,
    SpanQueryOrder, StaleSource, WriteWatchRefresh, WrittenRegion, create_mipmapped_guest_texture,
    decode_guest_texture_linear_from_raw, decode_mode27_mip_chain_from_raw,
    decode_sampled_texture_linear_from_raw, depth_sample_view_must_not_steal_depth_authority,
    guest_ranges_overlap, guest_texture_byte_count, guest_texture_descriptors_share_byte_storage,
    guest_texture_descriptors_share_color_view_storage,
    guest_texture_descriptors_share_gpu_sample_storage, guest_texture_descriptors_share_storage,
    initial_depth_clear_for_target, mode27_mip_chain_layout, new_color_alias_texture_view,
    promote_resident_mip_hierarchy, record_mode27_work_detail,
    refresh_resident_texture_write_watch, resident_texture_eviction_plan,
    resident_texture_for_sample, retire_overlapping_sampled_textures,
    retire_sampled_textures_overlapping_writeback, snapshot_watched_guest_range,
    snapshot_watched_guest_texture_raw, snapshot_watched_sampled_texture_raw, spans_overlap,
    texture_element_layout, tiled_block_bytes, tiled_block_dimensions, trace_residency_churn,
    trace_resident_texture_demotion, trace_resident_texture_removal, watch_guest_texture,
    watch_resident_texture, watch_sampled_texture, written_byte_ranges,
};
use metal::{
    Buffer, ComputePipelineState, Device, MTLBlitOption, MTLLoadAction, MTLOrigin, MTLRegion,
    MTLResourceOptions, MTLSize, MTLStorageMode, MTLStoreAction, MTLTextureType, MTLTextureUsage,
    RenderPassDescriptor, Texture, TextureDescriptor,
};
use remu_memory::{
    CpuWriteWatchRetirement, guest_shared_backing_identity, guest_shared_backing_range,
};
use std::collections::HashSet;
use std::ffi::c_void;
use std::sync::OnceLock;
use std::sync::atomic::Ordering;
use std::time::Instant;

/// A/B lever for the range algebra: `REMU_RANGE_ALGEBRA=0` makes every write claim its writer's
/// whole declared range, which supersedes exactly what the descriptor-overlap retirement did.
fn range_algebra_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| std::env::var("REMU_RANGE_ALGEBRA").as_deref() != Ok("0"))
}

/// A Metal texture together with the residency-map key the ensure path served it from.
///
/// The selected key is not always the requested descriptor: a GPU-written resident whose row
/// pitch differs is still allowed to serve the request. Marking that write against the requested
/// descriptor resolves no resident, so ownership, content version and activity are lost while
/// the sampled caches are retired anyway. Callers that later mark the target written must carry
/// `descriptor` back into the mark path.
#[derive(Clone, Debug)]
pub(crate) struct SelectedResidentTexture {
    pub(crate) texture: Texture,
    pub(crate) descriptor: GuestTextureDescriptor,
}

enum StaleReadRematerialisation {
    View(Texture),
    Blit {
        source: Texture,
        destination: Texture,
        source_origin_y: u64,
    },
    Detile {
        destination: Texture,
    },
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum RematerialisationFailureKind {
    Mapping,
    CpuGpuConflict,
    Layout,
    Metal,
}

fn record_rematerialisation_failure(kind: RematerialisationFailureKind) {
    OWNED_REMATERIALISE_FAILED.fetch_add(1, Ordering::Relaxed);
    match kind {
        RematerialisationFailureKind::Mapping => {
            OWNED_REMATERIALISE_FAILED_MAPPING.fetch_add(1, Ordering::Relaxed);
        }
        RematerialisationFailureKind::CpuGpuConflict => {
            OWNED_REMATERIALISE_FAILED_CPU_GPU.fetch_add(1, Ordering::Relaxed);
        }
        RematerialisationFailureKind::Layout => {
            OWNED_REMATERIALISE_FAILED_LAYOUT.fetch_add(1, Ordering::Relaxed);
        }
        RematerialisationFailureKind::Metal => {
            OWNED_REMATERIALISE_FAILED_METAL.fetch_add(1, Ordering::Relaxed);
        }
    }
}

#[derive(Clone, Copy, Debug)]
struct Mode27RematerialisationSubresource {
    width: u64,
    height: u64,
    linear_offset: u64,
    linear_row_bytes: u64,
    linear_image_bytes: u64,
    level_offset: u64,
    pitch: u64,
    bytes_per_element: u64,
    block_width: u64,
    block_height: u64,
    tail_x: u64,
    tail_y: u64,
    surface_z_xor: u64,
    in_tail: bool,
    metal_slice: u64,
    metal_level: u64,
    metal_z: u64,
    tile_mode: u64,
}

fn align_256(value: u64) -> Option<u64> {
    value.checked_add(255).map(|value| value & !255)
}

#[allow(clippy::too_many_lines)]
fn mode27_rematerialisation_subresources(
    resident: &ResidentGuestTexture,
) -> Result<(Vec<Mode27RematerialisationSubresource>, u64), String> {
    let tile_mode = resident.descriptor.tile_mode;
    let valid_element_size = match tile_mode {
        24 => matches!(resident.descriptor.bytes_per_pixel, 1 | 2 | 4 | 8),
        27 => matches!(resident.descriptor.bytes_per_pixel, 1 | 2 | 4 | 8 | 16),
        _ => false,
    };
    if !valid_element_size || resident.layer_count == 0 || resident.mip_level_count == 0 {
        return Err(format!(
            "GPU retile requires mode-24 1/2/4/8-byte or mode-27 1/2/4/8/16-byte storage, got mode={} bpe={} layers={} mips={}",
            tile_mode,
            resident.descriptor.bytes_per_pixel,
            resident.layer_count,
            resident.mip_level_count,
        ));
    }
    if resident.mip_level_count > 1
        && (tile_mode != 27 || resident.texture_type != 9 || resident.layer_count != 1)
    {
        return Err(format!(
            "GPU retile cannot represent mipmapped type={} layers={}",
            resident.texture_type, resident.layer_count
        ));
    }
    if !matches!(resident.texture_type, 9 | 10 | 11 | 13) {
        return Err(format!(
            "GPU retile cannot represent texture type {}",
            resident.texture_type
        ));
    }
    if super::super::metal_uncompressed_bytes_per_pixel(resident.descriptor.pixel_format)
        != Some(resident.descriptor.bytes_per_pixel)
    {
        return Err(format!(
            "GPU retile requires an exact uncompressed Metal layout for {:?}/{} bytes",
            resident.descriptor.pixel_format, resident.descriptor.bytes_per_pixel
        ));
    }
    let expected_type = match resident.texture_type {
        9 => MTLTextureType::D2,
        10 => MTLTextureType::D3,
        11 | 13 => MTLTextureType::D2Array,
        _ => unreachable!("texture type was validated"),
    };
    if resident.texture.texture_type() != expected_type
        || resident.texture.pixel_format() != resident.descriptor.pixel_format
        || resident.texture.sample_count() != 1
        || resident.texture.framebuffer_only()
        || resident.texture.storage_mode() == MTLStorageMode::Memoryless
        || resident.texture.width() < u64::from(resident.descriptor.width)
        || resident.texture.height() < u64::from(resident.descriptor.height)
        || resident.texture.mipmap_level_count() < u64::from(resident.mip_level_count)
        || (expected_type == MTLTextureType::D3
            && resident.texture.depth() < u64::from(resident.layer_count))
        || (expected_type == MTLTextureType::D2Array
            && resident.texture.array_length() < u64::from(resident.layer_count))
    {
        return Err(format!(
            "GPU retile Metal storage does not match type={} {}x{} layers={} mips={} {:?}",
            resident.texture_type,
            resident.descriptor.width,
            resident.descriptor.height,
            resident.layer_count,
            resident.mip_level_count,
            resident.descriptor.pixel_format,
        ));
    }
    let bytes_per_element = u64::from(resident.descriptor.bytes_per_pixel);
    let block_bytes = tiled_block_bytes(tile_mode).ok_or("tiled block size is unavailable")?;
    let (base_block_width, base_block_height) = tiled_block_dimensions(
        block_bytes,
        usize::try_from(bytes_per_element).map_err(|_| "element size does not fit usize")?,
    )
    .ok_or("mode-27 block dimensions are unavailable")?;
    let mut linear_offset = 0_u64;
    let mut result = Vec::new();
    let mut push = |width: u64,
                    height: u64,
                    level_offset: u64,
                    pitch: u64,
                    block_width: u64,
                    block_height: u64,
                    tail_x: u64,
                    tail_y: u64,
                    in_tail: bool,
                    surface_z: u32,
                    metal_level: u64|
     -> Result<(), String> {
        let packed_row = width
            .checked_mul(bytes_per_element)
            .ok_or("GPU retile packed row overflows")?;
        let linear_row_bytes = align_256(packed_row).ok_or("GPU retile row alignment overflows")?;
        let linear_image_bytes = linear_row_bytes
            .checked_mul(height)
            .ok_or("GPU retile image size overflows")?;
        linear_offset = align_256(linear_offset).ok_or("GPU retile offset overflows")?;
        let metal_slice = if matches!(resident.texture_type, 11 | 13) {
            u64::from(surface_z)
        } else {
            0
        };
        let metal_z = if resident.texture_type == 10 {
            u64::from(surface_z)
        } else {
            0
        };
        result.push(Mode27RematerialisationSubresource {
            width,
            height,
            linear_offset,
            linear_row_bytes,
            linear_image_bytes,
            level_offset,
            pitch,
            bytes_per_element,
            block_width,
            block_height,
            tail_x,
            tail_y,
            surface_z_xor: u64::try_from(super::mode27_surface_z_xor(
                usize::try_from(surface_z).expect("surface index fits usize"),
            ))
            .expect("surface xor fits u64"),
            in_tail,
            metal_slice,
            metal_level,
            metal_z,
            tile_mode: u64::from(tile_mode),
        });
        linear_offset = linear_offset
            .checked_add(linear_image_bytes)
            .ok_or("GPU retile linear buffer size overflows")?;
        Ok(())
    };
    if resident.mip_level_count > 1 {
        let layout = mode27_mip_chain_layout(resident.descriptor, resident.mip_level_count)?;
        for (level_index, level) in layout.levels.iter().enumerate() {
            push(
                u64::try_from(level.width).map_err(|_| "mip width does not fit u64")?,
                u64::try_from(level.height).map_err(|_| "mip height does not fit u64")?,
                u64::try_from(level.offset).map_err(|_| "mip offset does not fit u64")?,
                u64::try_from(level.padded_width).map_err(|_| "mip pitch does not fit u64")?,
                u64::try_from(layout.block_width)
                    .map_err(|_| "mip block width does not fit u64")?,
                u64::try_from(layout.block_height)
                    .map_err(|_| "mip block height does not fit u64")?,
                u64::try_from(level.tail_x).map_err(|_| "mip tail X does not fit u64")?,
                u64::try_from(level.tail_y).map_err(|_| "mip tail Y does not fit u64")?,
                level.in_tail,
                0,
                u64::try_from(level_index).map_err(|_| "mip index does not fit u64")?,
            )?;
        }
    } else {
        let face_bytes = guest_texture_byte_count(resident.descriptor)
            .ok_or("GPU retile face size overflows")?;
        for surface_z in 0..resident.layer_count {
            push(
                u64::from(resident.descriptor.width),
                u64::from(resident.descriptor.height),
                face_bytes
                    .checked_mul(u64::from(surface_z))
                    .ok_or("GPU retile layer offset overflows")?,
                u64::from(resident.descriptor.pitch),
                u64::try_from(base_block_width).map_err(|_| "block width does not fit u64")?,
                u64::try_from(base_block_height).map_err(|_| "block height does not fit u64")?,
                0,
                0,
                false,
                surface_z,
                0,
            )?;
        }
    }
    Ok((result, linear_offset))
}

struct Mode27RematerialisationInput {
    texture: Texture,
    span: ResourceSpan,
    debt: RangeSet,
    subresources: Vec<Mode27RematerialisationSubresource>,
    linear_length: u64,
}

fn mode27_rematerialisation_parameters(
    subresource: Mode27RematerialisationSubresource,
    resource_base: u64,
    target_span: ResourceSpan,
    run_count: usize,
) -> Result<[u64; 18], String> {
    Ok([
        subresource.width,
        subresource.height,
        subresource.linear_row_bytes,
        subresource.linear_offset,
        resource_base,
        target_span.range_start(),
        subresource.level_offset,
        subresource.pitch,
        subresource.bytes_per_element,
        subresource.block_width,
        subresource.block_height,
        subresource.tail_x,
        subresource.tail_y,
        subresource.surface_z_xor,
        u64::from(subresource.in_tail),
        target_span.len(),
        u64::try_from(run_count).map_err(|_| "GPU retile run count does not fit u64")?,
        subresource.tile_mode,
    ])
}

/// Returns `(retile/scatter, detile/gather)` pipelines from the S2 tiler module.
fn compile_mode27_rematerialisation_pipelines(
    device: &Device,
) -> Result<(ComputePipelineState, ComputePipelineState), String> {
    let (detile, retile) = mode27_tiler_pipelines(device)?;
    // Historical rematerialisation field order: scatter (=retile) then gather (=detile).
    Ok((retile, detile))
}

fn encode_texture_to_linear_buffer(
    blit: &metal::BlitCommandEncoderRef,
    texture: &Texture,
    linear: &Buffer,
    subresources: &[Mode27RematerialisationSubresource],
) {
    for subresource in subresources {
        blit.copy_from_texture_to_buffer(
            texture,
            subresource.metal_slice,
            subresource.metal_level,
            MTLOrigin {
                x: 0,
                y: 0,
                z: subresource.metal_z,
            },
            MTLSize::new(subresource.width, subresource.height, 1),
            linear,
            subresource.linear_offset,
            subresource.linear_row_bytes,
            subresource.linear_image_bytes,
            MTLBlitOption::None,
        );
    }
}

fn encode_linear_buffer_to_texture(
    blit: &metal::BlitCommandEncoderRef,
    linear: &Buffer,
    texture: &Texture,
    subresources: &[Mode27RematerialisationSubresource],
) {
    for subresource in subresources {
        blit.copy_from_buffer_to_texture(
            linear,
            subresource.linear_offset,
            subresource.linear_row_bytes,
            subresource.linear_image_bytes,
            MTLSize::new(subresource.width, subresource.height, 1),
            texture,
            subresource.metal_slice,
            subresource.metal_level,
            MTLOrigin {
                x: 0,
                y: 0,
                z: subresource.metal_z,
            },
            MTLBlitOption::None,
        );
    }
}

impl MetalPresenter {
    /// Reconstructs `target_descriptor` from RAM overlaid with every current GPU publication
    /// debt owner. All work stays in the open guest batch: the shared backing is read, never
    /// written, and no completion boundary is introduced.
    #[allow(clippy::too_many_lines)]
    fn encode_mode27_stale_rematerialisation(
        &mut self,
        target_descriptor: GuestTextureDescriptor,
        target_span: ResourceSpan,
        source_descriptors: &[GuestTextureDescriptor],
    ) -> Result<Texture, String> {
        let device = self
            .device
            .as_ref()
            .ok_or("Metal device is unavailable")?
            .clone();
        if self.mode27_rematerialisation_scatter_pipeline.is_none()
            || self.mode27_rematerialisation_gather_pipeline.is_none()
        {
            let (scatter, gather) = compile_mode27_rematerialisation_pipelines(&device)?;
            self.mode27_rematerialisation_scatter_pipeline = Some(scatter);
            self.mode27_rematerialisation_gather_pipeline = Some(gather);
        }
        let scatter_pipeline = self
            .mode27_rematerialisation_scatter_pipeline
            .as_ref()
            .expect("mode-27 scatter pipeline was initialized")
            .clone();
        let gather_pipeline = self
            .mode27_rematerialisation_gather_pipeline
            .as_ref()
            .expect("mode-27 gather pipeline was initialized")
            .clone();

        let target = self
            .resident_guest_textures
            .get(&target_descriptor)
            .ok_or("stale rematerialisation target disappeared")?;
        let destination = target.texture.clone();
        let (target_subresources, target_linear_length) =
            mode27_rematerialisation_subresources(target)?;
        let mut sources = Vec::with_capacity(source_descriptors.len());
        for descriptor in source_descriptors {
            let source = self
                .resident_guest_textures
                .get(descriptor)
                .ok_or("stale rematerialisation source disappeared")?;
            let span = source
                .span
                .ok_or("stale rematerialisation source has no allocation span")?;
            let debt = source
                .gpu_authored
                .intersect(&RangeSet::for_span(&target_span));
            if debt.is_empty() {
                continue;
            }
            let (subresources, linear_length) = mode27_rematerialisation_subresources(source)?;
            sources.push(Mode27RematerialisationInput {
                texture: source.texture.clone(),
                span,
                debt,
                subresources,
                linear_length,
            });
        }
        if sources.is_empty() {
            return Err("GPU retile was requested without a current debt owner".to_owned());
        }

        let backing_range = guest_shared_backing_range(target_span.address(), target_span.len())
            .ok_or("stale target has no contiguous guest backing view")?;
        let identity = backing_range.view_identity();
        if target_span.backing().is_none_or(|target| {
            target.backing_id != identity.backing_id
                || target.backing_offset
                    != identity
                        .backing_offset
                        .saturating_add(backing_range.resource_offset())
        }) {
            return Err("stale target backing view changed after residency admission".to_owned());
        }
        let guest = metal_buffer_with_guest_backing_range(&device, &backing_range)
            .ok_or("stale target guest backing cannot be exposed to Metal")?;
        let scratch = device.new_buffer(target_span.len(), MTLResourceOptions::StorageModePrivate);
        scratch.set_label("REMu mode-27 rematerialisation tiled scratch");
        let target_linear =
            device.new_buffer(target_linear_length, MTLResourceOptions::StorageModePrivate);
        target_linear.set_label("REMu mode-27 rematerialisation target linear");

        self.ensure_active_open_guest_batch()?;
        let command_buffer = {
            let batch = self
                .active_open_guest_batch_mut()
                .expect("guest batch was opened for texture rematerialisation");
            batch.close_active_pass_encoders();
            batch.command_buffer.clone()
        };

        let initialize = command_buffer.new_blit_command_encoder();
        initialize.set_label("REMu initialize rematerialisation from guest bytes");
        initialize.copy_from_buffer(
            &guest,
            backing_range.resource_offset(),
            &scratch,
            0,
            target_span.len(),
        );
        initialize.end_encoding();

        for source in &sources {
            let linear =
                device.new_buffer(source.linear_length, MTLResourceOptions::StorageModePrivate);
            linear.set_label("REMu mode-27 rematerialisation source linear");
            let blit = command_buffer.new_blit_command_encoder();
            blit.set_label("REMu linearize current texture owner");
            encode_texture_to_linear_buffer(blit, &source.texture, &linear, &source.subresources);
            blit.end_encoding();

            let mut run_words = Vec::with_capacity(source.debt.runs().len() * 2);
            for &(start, end) in source.debt.runs() {
                run_words.extend_from_slice(&[start, end]);
            }
            let run_bytes = u64::try_from(run_words.len())
                .ok()
                .and_then(|words| words.checked_mul(8))
                .ok_or("GPU retile run buffer size overflows")?;
            let runs = device.new_buffer_with_data(
                run_words.as_ptr().cast::<c_void>(),
                run_bytes,
                MTLResourceOptions::StorageModeShared,
            );
            runs.set_label("REMu mode-27 rematerialisation debt runs");
            let compute = command_buffer.new_compute_command_encoder();
            compute.set_label("REMu retile current texture owner");
            compute.set_compute_pipeline_state(&scatter_pipeline);
            compute.set_buffer(0, Some(&linear), 0);
            compute.set_buffer(1, Some(&scratch), 0);
            compute.set_buffer(2, Some(&runs), 0);
            for subresource in &source.subresources {
                let parameters = mode27_rematerialisation_parameters(
                    *subresource,
                    source.span.range_start(),
                    target_span,
                    source.debt.runs().len(),
                )?;
                compute.set_bytes(
                    3,
                    u64::try_from(std::mem::size_of_val(&parameters))
                        .expect("parameter byte count fits u64"),
                    parameters.as_ptr().cast::<c_void>(),
                );
                compute.dispatch_threads(
                    MTLSize::new(subresource.width, subresource.height, 1),
                    MTLSize::new(16, 16, 1),
                );
            }
            compute.end_encoding();
        }

        let gather = command_buffer.new_compute_command_encoder();
        gather.set_label("REMu detile stale target");
        gather.set_compute_pipeline_state(&gather_pipeline);
        gather.set_buffer(0, Some(&scratch), 0);
        gather.set_buffer(1, Some(&target_linear), 0);
        for subresource in &target_subresources {
            let parameters = mode27_rematerialisation_parameters(
                *subresource,
                target_span.range_start(),
                target_span,
                0,
            )?;
            gather.set_bytes(
                3,
                u64::try_from(std::mem::size_of_val(&parameters))
                    .expect("parameter byte count fits u64"),
                parameters.as_ptr().cast::<c_void>(),
            );
            gather.dispatch_threads(
                MTLSize::new(subresource.width, subresource.height, 1),
                MTLSize::new(16, 16, 1),
            );
        }
        gather.end_encoding();

        let upload = command_buffer.new_blit_command_encoder();
        upload.set_label("REMu upload detiled stale target");
        encode_linear_buffer_to_texture(upload, &target_linear, &destination, &target_subresources);
        upload.end_encoding();
        self.active_open_guest_batch_mut()
            .expect("rematerialisation batch remains open")
            .record_resident_write_footprint(
                SharedEventWriteKind::InternalResidentDestination,
                Some(target_span),
            );
        Ok(destination)
    }
}

/// Source-row offset for the byte-exact rematerialisation blit rung.
///
/// A linear surface may move by whole pitch rows. Mode 27 may move by whole block rows only when
/// the shift preserves the block-row parity that selects the within-block swizzle table. Other
/// tilings stay on the explicit fallback until their address transform has an equivalent proof.
pub(crate) fn rematerialisation_blit_source_origin_y(
    source: GuestTextureDescriptor,
    target: GuestTextureDescriptor,
    source_span: ResourceSpan,
    target_span: ResourceSpan,
) -> Option<u64> {
    if source.tile_mode != target.tile_mode
        || source.bytes_per_pixel != target.bytes_per_pixel
        || source.pitch != target.pitch
        || source.pixel_format != target.pixel_format
        || !source_span.shares_offset_space(&target_span)
    {
        return None;
    }
    let delta_bytes = target_span
        .range_start()
        .checked_sub(source_span.range_start())?;
    match source.tile_mode {
        0 => {
            let row_bytes =
                u64::from(source.pitch).checked_mul(u64::from(source.bytes_per_pixel))?;
            (row_bytes != 0 && delta_bytes % row_bytes == 0).then_some(delta_bytes / row_bytes)
        }
        27 => {
            let block_bytes = tiled_block_bytes(27)?;
            let bytes_per_pixel = usize::try_from(source.bytes_per_pixel).ok()?;
            let (block_width, block_height) = tiled_block_dimensions(block_bytes, bytes_per_pixel)?;
            let pitch = usize::try_from(source.pitch).ok()?;
            let blocks_per_row = pitch.div_ceil(block_width);
            let block_row_bytes = blocks_per_row.checked_mul(block_bytes)?;
            let block_row_bytes = u64::try_from(block_row_bytes).ok()?;
            if block_row_bytes == 0 || delta_bytes % block_row_bytes != 0 {
                return None;
            }
            let block_rows = delta_bytes / block_row_bytes;
            if block_rows % 2 != 0 {
                return None;
            }
            block_rows.checked_mul(u64::try_from(block_height).ok()?)
        }
        _ => None,
    }
}

/// Read-only attribution for persistent Metal render/storage targets.
///
/// GPU-owned residents may contain the only current copy of partially rendered pixels and cannot
/// be dropped merely to satisfy a host-memory budget. Mapping retirement and repeated physical
/// backing identities are the semantics-bearing signals needed to distinguish stale residents
/// from live render-graph allocations.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub(crate) struct ResidentTextureOwnershipStats {
    pub(crate) gpu_owned_count: usize,
    pub(crate) gpu_owned_bytes: u64,
    pub(crate) cpu_owned_count: usize,
    pub(crate) cpu_owned_bytes: u64,
    /// Demoted allocations, which answer for no byte but still pin their Metal storage.
    pub(crate) stale_count: usize,
    pub(crate) stale_bytes: u64,
    pub(crate) active_mapping_count: usize,
    pub(crate) active_mapping_bytes: u64,
    pub(crate) reprotected_mapping_count: usize,
    pub(crate) reprotected_mapping_bytes: u64,
    pub(crate) reprotecting_mapping_count: usize,
    pub(crate) reprotecting_mapping_bytes: u64,
    pub(crate) replaced_mapping_count: usize,
    pub(crate) replaced_mapping_bytes: u64,
    pub(crate) unique_backing_bases: usize,
    pub(crate) duplicate_backing_bases: usize,
    pub(crate) unique_backing_spans: usize,
    pub(crate) duplicate_backing_spans: usize,
    pub(crate) unresolved_backing_count: usize,
    pub(crate) unresolved_backing_bytes: u64,
}

pub(crate) fn resident_texture_ownership_stats(
    residents: &ResidentTextureMap,
) -> ResidentTextureOwnershipStats {
    let mut stats = ResidentTextureOwnershipStats::default();
    let mut backing_bases = HashSet::new();
    let mut backing_spans = HashSet::new();
    for resident in residents.values() {
        let allocated_bytes = resident.texture.allocated_size();
        if resident.gpu_written() {
            stats.gpu_owned_count += 1;
            stats.gpu_owned_bytes = stats.gpu_owned_bytes.saturating_add(allocated_bytes);
        } else {
            stats.cpu_owned_count += 1;
            stats.cpu_owned_bytes = stats.cpu_owned_bytes.saturating_add(allocated_bytes);
        }
        if !resident.is_live() {
            stats.stale_count += 1;
            stats.stale_bytes = stats.stale_bytes.saturating_add(allocated_bytes);
        }

        let mapping_is_live = match resident.write_watch.retirement() {
            None => {
                stats.active_mapping_count += 1;
                stats.active_mapping_bytes =
                    stats.active_mapping_bytes.saturating_add(allocated_bytes);
                true
            }
            Some(CpuWriteWatchRetirement::Reprotected) => {
                stats.reprotected_mapping_count += 1;
                stats.reprotected_mapping_bytes = stats
                    .reprotected_mapping_bytes
                    .saturating_add(allocated_bytes);
                true
            }
            Some(CpuWriteWatchRetirement::Reprotecting) => {
                stats.reprotecting_mapping_count += 1;
                stats.reprotecting_mapping_bytes = stats
                    .reprotecting_mapping_bytes
                    .saturating_add(allocated_bytes);
                true
            }
            Some(CpuWriteWatchRetirement::Replaced) => {
                stats.replaced_mapping_count += 1;
                stats.replaced_mapping_bytes =
                    stats.replaced_mapping_bytes.saturating_add(allocated_bytes);
                false
            }
        };
        if !mapping_is_live {
            continue;
        }
        let (address, byte_count) = resident.write_watch.range();
        let Some(identity) = guest_shared_backing_identity(address, byte_count) else {
            stats.unresolved_backing_count += 1;
            stats.unresolved_backing_bytes = stats
                .unresolved_backing_bytes
                .saturating_add(allocated_bytes);
            continue;
        };
        if !backing_bases.insert((identity.backing_id, identity.backing_offset)) {
            stats.duplicate_backing_bases += 1;
        }
        if !backing_spans.insert((identity.backing_id, identity.backing_offset, byte_count)) {
            stats.duplicate_backing_spans += 1;
        }
    }
    stats.unique_backing_bases = backing_bases.len();
    stats.unique_backing_spans = backing_spans.len();
    stats
}

impl MetalPresenter {
    /// Records the current resident as an internal-relayout version on the open batch
    /// (plan A2), so a later same-queue draw can sample it through the shared-event
    /// checkpoint instead of draining the host.
    fn record_internal_resident_version(&mut self, descriptor: GuestTextureDescriptor) {
        let Some(version) = self
            .resident_guest_textures
            .get(&descriptor)
            .and_then(|resident| {
                Some(SharedEventResidentTextureVersion {
                    texture: resident.texture.clone(),
                    descriptor: resident.descriptor,
                    texture_type: resident.texture_type,
                    layer_count: resident.layer_count,
                    mip_level_count: resident.mip_level_count,
                    content_version: resident.content_version,
                    span: resident.span?,
                    kind: SharedEventWriteKind::InternalResidentDestination,
                })
            })
        else {
            return;
        };
        if let Some(batch) = self.active_open_guest_batch_mut() {
            batch.record_resident_texture_version(version);
        }
    }
}

impl MetalPresenter {
    /// Restores a demoted read target from the allocation that currently owns its bytes.
    ///
    /// View reuse is immediate. A copy is appended to the existing guest command buffer after
    /// closing its active encoders, which orders it after the producer and before the consumer
    /// without committing, flushing, or waiting. Returning `false` is permitted only when guest
    /// RAM is provably authoritative; GPU-current failures stay stale and are reported loudly.
    #[allow(clippy::too_many_lines)]
    pub(crate) fn rematerialise_stale_resident_for_read(
        &mut self,
        target_descriptor: GuestTextureDescriptor,
    ) -> Result<bool, String> {
        let failure = |kind: RematerialisationFailureKind, reason: String| {
            record_rematerialisation_failure(kind);
            Err(reason)
        };

        let Some(target_state) = self
            .resident_guest_textures
            .get(&target_descriptor)
            .map(|resident| resident.state)
        else {
            return Ok(false);
        };
        let ResidencyState::Stale(stale_source) = target_state else {
            return Ok(false);
        };

        let target_refresh = refresh_resident_texture_write_watch(
            self.resident_guest_textures
                .get_mut(&target_descriptor)
                .expect("stale target still exists"),
        )?;
        if target_refresh == WriteWatchRefresh::Replaced {
            // The descriptor now names a different mapping. The ordinary ensure path creates a
            // fresh resident from that mapping; the old allocation relation is no longer valid.
            OWNED_REMATERIALISE_CPU.fetch_add(1, Ordering::Relaxed);
            return Ok(false);
        }
        if target_refresh == WriteWatchRefresh::Reprotecting {
            return failure(
                RematerialisationFailureKind::Mapping,
                format!(
                    "cannot rematerialise stale texture {:#x} while its mapping is being reprotected",
                    target_descriptor.address
                ),
            );
        }

        let target_cpu_may_have_changed = self
            .resident_guest_textures
            .get(&target_descriptor)
            .is_some_and(|resident| {
                matches!(
                    target_refresh,
                    WriteWatchRefresh::Renewed { was_dirty: true }
                ) || resident.write_watch.content_validation_pending()
                    || resident.write_watch.is_dirty()
            });
        let target_cpu_changed = target_cpu_may_have_changed
            && self
                .resident_guest_textures
                .get(&target_descriptor)
                .expect("stale target still exists")
                .write_watch
                .content_has_changed()?;

        let StaleSource::Allocation { .. } = stale_source else {
            OWNED_REMATERIALISE_CPU.fetch_add(1, Ordering::Relaxed);
            return Ok(false);
        };

        let Some((target_span, target_texture_type, target_layers, target_mips)) = self
            .resident_guest_textures
            .get(&target_descriptor)
            .and_then(|target| {
                Some((
                    target.span?,
                    target.texture_type,
                    target.layer_count,
                    target.mip_level_count,
                ))
            })
        else {
            return failure(
                RematerialisationFailureKind::Mapping,
                format!(
                    "stale texture {:#x} has no resolved allocation span",
                    target_descriptor.address
                ),
            );
        };
        let moved = RangeSet::for_span(&target_span);

        // A stale link records the owner at demotion time, but authority can subsequently move
        // through several pooled tenants. Resolve the current disjoint owners from `owned`
        // instead of following an obsolete descriptor/version chain.
        let candidates = self
            .resident_guest_textures
            .overlapping(&target_span, SpanQueryOrder::ResidentFirst);
        for candidate in &candidates {
            if *candidate == target_descriptor {
                continue;
            }
            let refresh = refresh_resident_texture_write_watch(
                self.resident_guest_textures
                    .get_mut(candidate)
                    .expect("indexed rematerialisation candidate exists"),
            )?;
            if matches!(
                refresh,
                WriteWatchRefresh::Replaced | WriteWatchRefresh::Reprotecting
            ) {
                return failure(
                    RematerialisationFailureKind::Mapping,
                    format!(
                        "current texture owner {:#x} changed mapping while restoring stale texture {:#x}",
                        candidate.address, target_descriptor.address
                    ),
                );
            }
        }
        let mut owners = Vec::new();
        let mut authority = RangeSet::default();
        let mut debt = RangeSet::default();
        let mut full_owner = None;
        for candidate in candidates {
            if candidate == target_descriptor {
                continue;
            }
            let Some(owner) = self.resident_guest_textures.get(&candidate) else {
                continue;
            };
            let Some(owner_span) = owner.span else {
                continue;
            };
            if !owner.is_live() || !owner_span.shares_offset_space(&target_span) {
                continue;
            }
            let held = owner.owned.intersect(&moved);
            if held.is_empty() {
                continue;
            }
            if authority.intersects(&held) {
                return failure(
                    RematerialisationFailureKind::Mapping,
                    format!(
                        "overlapping current owners reached stale texture {:#x}",
                        target_descriptor.address
                    ),
                );
            }
            authority.insert(&held);
            let owner_debt = owner.gpu_authored.intersect(&moved);
            if !owner_debt.is_empty() {
                debt.insert(&owner_debt);
                owners.push(candidate);
            }
            if owner.owned.contains_all(&moved) {
                full_owner = Some(candidate);
            }
        }
        if target_cpu_changed {
            if debt.is_empty() {
                self.note_guest_memory_authoritative(
                    &target_span,
                    &moved,
                    None,
                    "rematerialise-clean-cpu-write",
                );
                self.resident_guest_textures
                    .get_mut(&target_descriptor)
                    .expect("stale target still exists")
                    .state = ResidencyState::Stale(StaleSource::GuestMemory);
                OWNED_REMATERIALISE_CPU.fetch_add(1, Ordering::Relaxed);
                return Ok(false);
            }
            return failure(
                RematerialisationFailureKind::CpuGpuConflict,
                format!(
                    "stale texture {:#x} has both CPU-modified bytes and GPU-only allocation debt",
                    target_descriptor.address
                ),
            );
        }
        // RAM is authoritative for every requested byte not present in this union. If no byte
        // exists solely in Metal, the ordinary validated rebuild remains the cheapest rung.
        if debt.is_empty() {
            self.note_guest_memory_authoritative(
                &target_span,
                &moved,
                None,
                "rematerialise-clean-guest-fallback",
            );
            self.resident_guest_textures
                .get_mut(&target_descriptor)
                .expect("stale target still exists")
                .state = ResidencyState::Stale(StaleSource::GuestMemory);
            OWNED_REMATERIALISE_CPU.fetch_add(1, Ordering::Relaxed);
            return Ok(false);
        }

        let mut rematerialisation = None;
        let mut source_version = None;
        if let Some(source_descriptor) = full_owner {
            let source = self
                .resident_guest_textures
                .get(&source_descriptor)
                .expect("resolved full owner exists");
            let source_span = source.span.expect("resolved full owner has a span");
            let same_host_shape = source.texture_type == target_texture_type
                && source.layer_count == target_layers
                && source.mip_level_count == target_mips;
            let view = if same_host_shape
                && guest_texture_descriptors_share_storage(source_descriptor, target_descriptor)
            {
                Some(source.texture.clone())
            } else if same_host_shape
                && guest_texture_descriptors_share_color_view_storage(
                    source_descriptor,
                    target_descriptor,
                )
            {
                new_color_alias_texture_view(
                    &source.texture,
                    target_descriptor.pixel_format,
                    0,
                    target_mips,
                )
            } else {
                None
            };
            if let Some(view) = view {
                rematerialisation = Some(StaleReadRematerialisation::View(view));
                source_version = Some(source.content_version);
            } else {
                let source_origin_y = (target_texture_type == 9
                    && target_layers == 1
                    && target_mips == 1
                    && source.texture_type == 9
                    && source.layer_count == 1
                    && source.mip_level_count == 1)
                    .then(|| {
                        rematerialisation_blit_source_origin_y(
                            source_descriptor,
                            target_descriptor,
                            source_span,
                            target_span,
                        )
                    })
                    .flatten()
                    .filter(|source_origin_y| {
                        source.texture.width() >= u64::from(target_descriptor.width)
                            && source
                                .texture
                                .height()
                                .checked_sub(*source_origin_y)
                                .is_some_and(|height| height >= u64::from(target_descriptor.height))
                    });
                if let Some(source_origin_y) = source_origin_y {
                    let destination = self
                        .resident_guest_textures
                        .get(&target_descriptor)
                        .expect("stale target still exists")
                        .texture
                        .clone();
                    if destination.width() >= u64::from(target_descriptor.width)
                        && destination.height() >= u64::from(target_descriptor.height)
                        && destination.pixel_format() == target_descriptor.pixel_format
                        && source.texture.pixel_format() == target_descriptor.pixel_format
                    {
                        rematerialisation = Some(StaleReadRematerialisation::Blit {
                            source: source.texture.clone(),
                            destination,
                            source_origin_y,
                        });
                        source_version = Some(source.content_version);
                    }
                }
            }
        }

        let rematerialisation = if let Some(rematerialisation) = rematerialisation {
            rematerialisation
        } else {
            let destination = match self.encode_mode27_stale_rematerialisation(
                target_descriptor,
                target_span,
                &owners,
            ) {
                Ok(destination) => destination,
                Err(error) => {
                    let kind = if error.contains("backing") || error.contains("mapping") {
                        RematerialisationFailureKind::Mapping
                    } else if error.contains("Metal") {
                        RematerialisationFailureKind::Metal
                    } else {
                        RematerialisationFailureKind::Layout
                    };
                    return failure(
                        kind,
                        format!(
                            "GPU-current stale texture {:#x} cannot be rematerialised from {} current owners: {error}",
                            target_descriptor.address,
                            owners.len()
                        ),
                    );
                }
            };
            self.resident_content_version = self.resident_content_version.saturating_add(1);
            source_version = Some(self.resident_content_version);
            StaleReadRematerialisation::Detile { destination }
        };

        if let StaleReadRematerialisation::Blit {
            source,
            destination,
            source_origin_y,
        } = &rematerialisation
        {
            self.ensure_active_open_guest_batch()?;
            let command_buffer = {
                let batch = self
                    .active_open_guest_batch_mut()
                    .expect("guest batch was opened for texture rematerialisation");
                batch.close_active_pass_encoders();
                batch.command_buffer.clone()
            };
            let blit = command_buffer.new_blit_command_encoder();
            blit.set_label("REMu rematerialise stale guest texture");
            blit.copy_from_texture(
                source,
                0,
                0,
                MTLOrigin {
                    x: 0,
                    y: *source_origin_y,
                    z: 0,
                },
                MTLSize::new(
                    u64::from(target_descriptor.width),
                    u64::from(target_descriptor.height),
                    1,
                ),
                destination,
                0,
                0,
                MTLOrigin { x: 0, y: 0, z: 0 },
            );
            blit.end_encoding();
            self.active_open_guest_batch_mut()
                .expect("rematerialisation batch remains open")
                .record_resident_write_footprint(
                    SharedEventWriteKind::InternalResidentDestination,
                    Some(target_span),
                );
        }

        // The transfer becomes visible to the residency algebra only after view creation or blit
        // encoding succeeds. Publication debt follows the same exact byte range.
        let source_version = source_version.expect("every rematerialisation has a content version");
        self.subtract_resident_authority(
            &target_span,
            &moved,
            Some(target_descriptor),
            StaleSource::Allocation {
                descriptor: target_descriptor,
                content_version: source_version,
            },
            "rematerialised-into-alias",
        );
        let mut target = self
            .resident_guest_textures
            .remove(&target_descriptor)
            .expect("validated stale target still exists");
        target.texture = match rematerialisation {
            StaleReadRematerialisation::View(view) => {
                OWNED_REMATERIALISE_VIEW.fetch_add(1, Ordering::Relaxed);
                view
            }
            StaleReadRematerialisation::Blit { destination, .. } => {
                OWNED_REMATERIALISE_BLIT.fetch_add(1, Ordering::Relaxed);
                destination
            }
            StaleReadRematerialisation::Detile { destination } => {
                OWNED_REMATERIALISE_DETILE.fetch_add(1, Ordering::Relaxed);
                destination
            }
        };
        target.component_reorder_view = None;
        target.owned = moved;
        target.gpu_authored = debt;
        target.state = ResidencyState::Live;
        target.content_version = source_version;
        if target.gpu_written() {
            target.note_gpu_write();
        }
        self.resident_guest_textures
            .insert(target_descriptor, target);
        self.record_internal_resident_version(target_descriptor);
        Ok(true)
    }

    pub(crate) fn garbage_collect_resident_textures_to_budget(
        &mut self,
        byte_capacity: u64,
        current_flip: u64,
        maximum_evictions: usize,
    ) -> Result<(usize, u64), String> {
        let plan = resident_texture_eviction_plan(
            &self.resident_guest_textures,
            byte_capacity,
            current_flip,
            maximum_evictions,
        );
        if plan.descriptors.is_empty() {
            return Ok((0, 0));
        }

        // End every encoder before observing or replacing resident allocations. Command buffers
        // retain their bound textures until completion, and the single Metal queue orders the
        // following readbacks after all producer work.
        self.wait_for_guest_command_batches()?;

        let mut eviction_count = 0_usize;
        let mut eviction_bytes = 0_u64;
        for descriptor in plan.descriptors {
            let Some(resident) = self.resident_guest_textures.get_mut(&descriptor) else {
                // Writing back an older overlapping version may already have retired this alias.
                continue;
            };
            let allocated_bytes = resident.texture.allocated_size();
            let refresh = refresh_resident_texture_write_watch(resident)?;
            if refresh == WriteWatchRefresh::Reprotecting {
                // A concurrent CPU write is changing watch ownership. Leave this resident in
                // place and retry at a later flip rather than publishing pixels into that range.
                continue;
            }
            if refresh == WriteWatchRefresh::Replaced {
                if let Some(resident) = self.resident_guest_textures.remove(&descriptor) {
                    trace_resident_texture_removal("pressure-mapping-replaced", &resident);
                }
                eviction_count += 1;
                eviction_bytes = eviction_bytes.saturating_add(allocated_bytes);
                continue;
            }

            let gpu_written = resident.gpu_written();
            if gpu_written && resident.cpu_write_obsoletes_gpu_writeback(refresh)? {
                if let Some(resident) = self.resident_guest_textures.remove(&descriptor) {
                    trace_resident_texture_removal("pressure-cpu-reused", &resident);
                }
                eviction_count += 1;
                eviction_bytes = eviction_bytes.saturating_add(allocated_bytes);
                continue;
            }

            if gpu_written {
                self.settle_resident_texture_writeback(
                    descriptor,
                    WritebackTrigger::PressureEviction,
                )?;
            }

            if let Some(resident) = self.resident_guest_textures.remove(&descriptor) {
                trace_resident_texture_removal("pressure-lru", &resident);
            }
            eviction_count += 1;
            eviction_bytes = eviction_bytes.saturating_add(allocated_bytes);
        }
        if eviction_count != 0 {
            eprintln!(
                "[REMu][METAL] resident texture pressure GC evicted {eviction_count} images/{:.1} MiB (resident {:.1} MiB, budget {:.1} MiB)",
                eviction_bytes as f64 / (1024.0 * 1024.0),
                plan.resident_bytes as f64 / (1024.0 * 1024.0),
                byte_capacity as f64 / (1024.0 * 1024.0),
            );
        }
        Ok((eviction_count, eviction_bytes))
    }

    pub(crate) fn garbage_collect_resident_textures(&mut self, current_flip: u64) {
        let Some(byte_capacity) = super::super::resident_texture_cache_byte_capacity() else {
            return;
        };
        if let Err(error) = self.garbage_collect_resident_textures_to_budget(
            byte_capacity,
            current_flip,
            super::super::MAX_RESIDENT_TEXTURE_EVICTIONS_PER_FLIP,
        ) {
            eprintln!("[REMu][METAL] resident texture pressure GC failed: {error}");
        }
    }

    pub(crate) fn ensure_mipmapped_storage_texture(
        &mut self,
        device: &Device,
        descriptor: DecodedImageDescriptor,
    ) -> Result<Texture, String> {
        self.rematerialise_stale_resident_for_read(descriptor.guest)?;
        let underlying_mip_count = descriptor
            .base_mip_level
            .checked_add(descriptor.mip_level_count)
            .ok_or("storage-image mip range overflows")?;
        let needs_gpu_promotion = self
            .resident_guest_textures
            .get(&descriptor.guest)
            .is_some_and(|resident| {
                resident.gpu_written()
                    && resident.texture_type == 9
                    && resident.layer_count == 1
                    && resident.mip_level_count < underlying_mip_count
            });
        if needs_gpu_promotion {
            self.ensure_active_open_guest_batch()?;
            let command_buffer = {
                let batch = self
                    .active_open_guest_batch_mut()
                    .expect("guest batch was opened for mip promotion");
                batch.close_active_pass_encoders();
                batch.command_buffer.clone()
            };
            let refresh = refresh_resident_texture_write_watch(
                self.resident_guest_textures
                    .get_mut(&descriptor.guest)
                    .expect("selected GPU-written resident exists"),
            )?;
            if refresh != WriteWatchRefresh::Replaced {
                let resident = self
                    .resident_guest_textures
                    .remove(&descriptor.guest)
                    .expect("selected GPU-written resident exists");
                let promoted = promote_resident_mip_hierarchy(
                    device,
                    &command_buffer,
                    descriptor,
                    resident,
                    Mode27WorkOrigin::StorageCreate,
                )?;
                self.resident_guest_textures
                    .insert(descriptor.guest, promoted);
                self.active_open_guest_batch_mut()
                    .expect("mip-promotion batch remains open")
                    .record_resident_write_footprint(
                        SharedEventWriteKind::InternalResidentDestination,
                        descriptor.span,
                    );
                self.record_internal_resident_version(descriptor.guest);
                return resident_texture_for_sample(
                    self.resident_guest_textures
                        .get_mut(&descriptor.guest)
                        .expect("promoted storage hierarchy exists"),
                    descriptor,
                )
                .ok_or_else(|| "promoted storage hierarchy cannot represent view".to_owned());
            }
            if let Some(resident) = self.resident_guest_textures.remove(&descriptor.guest) {
                trace_resident_texture_removal("mip-promotion-mapping-replaced", &resident);
            }
        }
        let needs_clean_promotion = self
            .resident_guest_textures
            .get(&descriptor.guest)
            .is_some_and(|resident| {
                !resident.gpu_written()
                    && resident.is_live()
                    && resident.texture_type == 9
                    && resident.layer_count == 1
                    && resident.mip_level_count < underlying_mip_count
            });
        let promotion_command_buffer = if needs_clean_promotion {
            self.ensure_active_open_guest_batch()?;
            let batch = self
                .active_open_guest_batch_mut()
                .expect("guest batch was opened for mip promotion");
            batch.end_compute_pass();
            batch.close_active_pass_encoders();
            Some(batch.command_buffer.clone())
        } else {
            None
        };
        let queue = self.command_queue.as_ref();
        let texture = super::ensure_mipmapped_resident_texture_in_map(
            device,
            queue,
            promotion_command_buffer.as_deref(),
            descriptor,
            &mut self.resident_guest_textures,
            Mode27WorkOrigin::StorageCreate,
        )?;
        if promotion_command_buffer.is_some() {
            self.active_open_guest_batch_mut()
                .expect("mip-promotion batch remains open")
                .record_resident_write_footprint(
                    SharedEventWriteKind::InternalResidentDestination,
                    descriptor.span,
                );
            self.record_internal_resident_version(descriptor.guest);
        }
        Ok(texture)
    }

    /// Ensures a resident for a descriptor no decoded structure carries a span for.
    ///
    /// Every guest-driven path reaches residency through a decoded descriptor and passes the span
    /// decode already resolved, so this is a test entry point.
    #[cfg(test)]
    pub(crate) fn ensure_resident_texture(
        &mut self,
        device: &Device,
        descriptor: GuestTextureDescriptor,
    ) -> Result<SelectedResidentTexture, String> {
        let span = ResourceSpan::for_texture(descriptor);
        self.ensure_resident_texture_with_initial_contents(device, descriptor, span, true)
    }

    pub(crate) fn ensure_resident_texture_with_initial_contents(
        &mut self,
        device: &Device,
        descriptor: GuestTextureDescriptor,
        span: Option<ResourceSpan>,
        upload_guest_contents: bool,
    ) -> Result<SelectedResidentTexture, String> {
        if upload_guest_contents && self.rematerialise_stale_resident_for_read(descriptor)? {
            return Ok(SelectedResidentTexture {
                texture: self
                    .resident_guest_textures
                    .get(&descriptor)
                    .expect("rematerialised resident exists")
                    .texture
                    .clone(),
                descriptor,
            });
        }
        if let Some(resident) =
            self.resident_guest_textures
                .get_mut(&descriptor)
                .filter(|resident| {
                    resident.texture_type == 9
                        && resident.layer_count == 1
                        && resident.mip_level_count >= 1
                })
        {
            let refresh = refresh_resident_texture_write_watch(resident)?;
            if resident.gpu_content_executed
                && resident.descriptor.width >= 1_280
                && resident.descriptor.height >= 720
            {
                static LOGGED: std::sync::atomic::AtomicUsize =
                    std::sync::atomic::AtomicUsize::new(0);
                let sequence = LOGGED.fetch_add(1, std::sync::atomic::Ordering::Relaxed) + 1;
                if sequence <= 16 || sequence.is_power_of_two() {
                    eprintln!(
                        "[REMu][METAL][ENSURE-REFRESH] sequence={sequence} address={:#x} refresh={refresh:?} upload_guest_contents={upload_guest_contents} version={}",
                        resident.descriptor.address, resident.content_version,
                    );
                }
            }
            match refresh {
                WriteWatchRefresh::Active
                | WriteWatchRefresh::Reprotecting
                | WriteWatchRefresh::Renewed { .. } => {
                    // A demoted target the guest is about to overwrite costs one state flip: no
                    // snapshot, no de-tile, no reprotect, no allocation. This is the path Unreal's
                    // transient allocator takes every time it re-binds a pooled range.
                    if resident.reclaim_for_bind(!upload_guest_contents) {
                        return Ok(SelectedResidentTexture {
                            texture: resident.texture.clone(),
                            descriptor,
                        });
                    }
                }
                WriteWatchRefresh::Replaced => {}
            }
        }
        if let Some(resident) = self.resident_guest_textures.remove(&descriptor) {
            if resident.gpu_content_executed
                && resident.descriptor.width >= 1_280
                && resident.descriptor.height >= 720
            {
                static LOGGED: std::sync::atomic::AtomicUsize =
                    std::sync::atomic::AtomicUsize::new(0);
                let sequence = LOGGED.fetch_add(1, std::sync::atomic::Ordering::Relaxed) + 1;
                if sequence <= 16 || sequence.is_power_of_two() {
                    eprintln!(
                        "[REMu][METAL][ENSURE-REPLACE] sequence={sequence} address={:#x} {}x{} {:?} version={} acquire_invalidated={} watch_retirement={:?}",
                        resident.descriptor.address,
                        resident.descriptor.width,
                        resident.descriptor.height,
                        resident.descriptor.pixel_format,
                        resident.content_version,
                        resident.acquire_invalidated,
                        resident.write_watch.retirement(),
                    );
                }
            }
            trace_resident_texture_removal("ensure-exact-resident-replaced", &resident);
        }
        let shared_resident = self
            .resident_guest_textures
            .values()
            .find(|resident| {
                resident.serves_reads()
                    && resident.texture_type == 9
                    && resident.layer_count == 1
                    && resident.mip_level_count >= 1
                    && guest_texture_descriptors_share_storage(resident.descriptor, descriptor)
            })
            .map(|resident| resident.descriptor);
        if let Some(shared_resident) = shared_resident {
            let resident = self
                .resident_guest_textures
                .get_mut(&shared_resident)
                .expect("selected shared resident texture exists");
            match refresh_resident_texture_write_watch(resident)? {
                WriteWatchRefresh::Active
                | WriteWatchRefresh::Reprotecting
                | WriteWatchRefresh::Renewed { .. } => {
                    return Ok(SelectedResidentTexture {
                        texture: resident.texture.clone(),
                        descriptor: shared_resident,
                    });
                }
                WriteWatchRefresh::Replaced => {
                    if let Some(resident) = self.resident_guest_textures.remove(&shared_resident) {
                        trace_resident_texture_removal(
                            "ensure-shared-resident-mapping-replaced",
                            &resident,
                        );
                    }
                }
            }
        }
        if let Some(gpu_sample_resident) = self
            .resident_guest_textures
            .values()
            .filter(|resident| {
                resident.gpu_written()
                    && resident.serves_reads()
                    && resident.texture_type == 9
                    && resident.layer_count == 1
                    && resident.mip_level_count >= 1
                    && guest_texture_descriptors_share_gpu_sample_storage(
                        resident.descriptor,
                        descriptor,
                    )
            })
            .max_by_key(|resident| resident.content_version)
            .map(|resident| resident.descriptor)
        {
            let resident = self
                .resident_guest_textures
                .get(&gpu_sample_resident)
                .expect("selected GPU sample resident exists");
            return Ok(SelectedResidentTexture {
                texture: resident.texture.clone(),
                descriptor: gpu_sample_resident,
            });
        }
        let mut aliases = span
            .iter()
            .flat_map(|span| {
                self.resident_guest_textures
                    .overlapping(span, SpanQueryOrder::ResidentFirst)
            })
            .filter_map(|alias| {
                // A demoted alias owes no writeback; its watch is refreshed by the acquire scan
                // and by the exact-key bind that reclaims or rebuilds it.
                self.resident_guest_textures
                    .get(&alias)
                    .filter(|resident| resident.is_live())
                    .map(|resident| (resident.content_version, resident.descriptor))
            })
            .collect::<Vec<_>>();
        if !aliases.is_empty() {
            trace_residency_churn(
                descriptor,
                upload_guest_contents,
                aliases.iter().filter_map(|(_, alias)| {
                    self.resident_guest_textures.get(alias).map(|resident| {
                        (
                            resident.descriptor,
                            resident.gpu_written(),
                            resident.content_version,
                        )
                    })
                }),
            );
            let has_gpu_written_alias = aliases.iter().any(|(_, alias)| {
                self.resident_guest_textures
                    .get(alias)
                    .is_some_and(|resident| resident.gpu_written())
            });
            if has_gpu_written_alias && WritebackTrigger::AliasRebuild.publishes_to_guest() {
                // The loop below downloads each GPU-written alias into guest memory; the download
                // is ordered behind its producer only once that producer is committed. A rebuild
                // that publishes nothing needs neither the drain nor the resolve invalidation.
                self.flush_guest_command_batch(
                    FlushPolicy::Required,
                    GuestBatchFlushReason::CpuSynchronization,
                )?;
                self.multisample_textures.retain(|key, _| {
                    !spans_overlap(ResourceSpan::for_texture(key.descriptor), span)
                });
            }
            aliases.sort_by_key(|(version, _)| *version);
            for (_, alias) in aliases {
                let refresh = refresh_resident_texture_write_watch(
                    self.resident_guest_textures
                        .get_mut(&alias)
                        .expect("collected resident alias exists"),
                )?;
                let gpu_written = self
                    .resident_guest_textures
                    .get(&alias)
                    .is_some_and(|resident| resident.gpu_written());
                let cpu_write_obsoletes_writeback = self
                    .resident_guest_textures
                    .get_mut(&alias)
                    .expect("refreshed resident alias exists")
                    .cpu_write_obsoletes_gpu_writeback(refresh)?;
                if refresh == WriteWatchRefresh::Reprotecting && gpu_written {
                    return Err(format!(
                        "resident alias mapping {:#x} is being reprotected",
                        alias.address
                    ));
                }
                if refresh == WriteWatchRefresh::Replaced {
                    if let Some(resident) = self.resident_guest_textures.remove(&alias) {
                        trace_resident_texture_removal("alias-mapping-replaced", &resident);
                    }
                } else if gpu_written {
                    if cpu_write_obsoletes_writeback {
                        // The guest reused this heap range after the last render. Publishing the
                        // old Metal image now would overwrite the new indirect-command bytes.
                        if let Some(resident) = self.resident_guest_textures.remove(&alias) {
                            trace_resident_texture_removal("cpu-reused-gpu-alias", &resident);
                        }
                        continue;
                    }
                    // A rebuild of an overlapping resource is REmu's own decision, not a guest
                    // read of these bytes: the guest never asked for this allocation's pixels to
                    // appear in RAM, and on hardware they were never staged there separately.
                    self.settle_resident_texture_writeback(alias, WritebackTrigger::AliasRebuild)?;
                }
            }
        }
        // The map path only ever adopts, reuses or creates under the exact key.
        Ok(SelectedResidentTexture {
            texture: self.ensure_resident_texture_in_map_phase1g(
                device,
                descriptor,
                upload_guest_contents,
            )?,
            descriptor,
        })
    }

    /// Phase 1G: create/adopt a resident under the exact key using GPU-timeline upload when
    /// the descriptor is tiler-owned. Never CPU-snapshots guest RAM for those modes.
    fn ensure_resident_texture_in_map_phase1g(
        &mut self,
        device: &Device,
        descriptor: GuestTextureDescriptor,
        upload_guest_contents: bool,
    ) -> Result<Texture, String> {
        if let Some(resident) =
            self.resident_guest_textures
                .get_mut(&descriptor)
                .filter(|resident| {
                    resident.texture_type == 9
                        && resident.layer_count == 1
                        && resident.mip_level_count >= 1
                })
        {
            match refresh_resident_texture_write_watch(resident)? {
                WriteWatchRefresh::Active
                | WriteWatchRefresh::Reprotecting
                | WriteWatchRefresh::Renewed { .. } => {
                    if resident.reclaim_for_bind(!upload_guest_contents) {
                        return Ok(resident.texture.clone());
                    }
                }
                WriteWatchRefresh::Replaced => {}
            }
        }
        if let Some(resident) = self.resident_guest_textures.remove(&descriptor) {
            trace_resident_texture_removal("ensure-exact-resident-replaced-phase1g", &resident);
        }
        let write_watch = watch_guest_texture(descriptor)?;
        let usage = MTLTextureUsage::RenderTarget
            | MTLTextureUsage::ShaderRead
            | MTLTextureUsage::ShaderWrite
            | MTLTextureUsage::PixelFormatView;
        let texture = if upload_guest_contents {
            if !gpu_tiler_owns_descriptor(descriptor) {
                return Err(format!(
                    "Phase 1G: refusing CPU snapshot upload for tile={} format={:?} bpe={} — no GPU tiler ownership",
                    descriptor.tile_mode, descriptor.pixel_format, descriptor.bytes_per_pixel
                ));
            }
            let texture = create_uninitialized_guest_texture(device, descriptor, usage)?;
            // Create-time upload uses a dedicated command buffer that settles before return so
            // a later CPU `replace_region` (tests) or GPU draw cannot race a deferred zero
            // upload still pending in the open guest batch. Steady-state publication stays on
            // the open guest batch via `encode_gpu_publish_to_guest`.
            let queue = self
                .command_queue
                .as_ref()
                .ok_or("Metal command queue is unavailable for Phase 1G upload")?
                .clone();
            let upload_buffer = queue.new_command_buffer().to_owned();
            upload_buffer.set_label("REMu Phase 1G create-time GPU upload");
            encode_gpu_upload_from_guest(device, &upload_buffer, descriptor, &texture)?;
            upload_buffer.commit();
            upload_buffer.wait_until_completed();
            texture
        } else {
            // Discarded targets still need a coarse watch; content baseline fingerprints are
            // deleted with the CPU upload path — the watch itself invalidates on guest writes.
            create_uninitialized_guest_texture(device, descriptor, usage)?
        };
        texture.set_label(&format!("REMu guest image {:#x}", descriptor.address));
        let (span, owned) = ResidentGuestTexture::declared_authority(&write_watch);
        self.resident_guest_textures.insert(
            descriptor,
            ResidentGuestTexture {
                texture: texture.clone(),
                component_reorder_view: None,
                descriptor,
                texture_type: 9,
                layer_count: 1,
                mip_level_count: 1,
                write_watch,
                span,
                owned,
                gpu_authored: RangeSet::default(),
                gpu_content_executed: false,
                state: ResidencyState::Live,
                acquire_invalidated: false,
                content_version: 0,
                gpu_write_watch_generations: Vec::new(),
                last_use: ResidentTextureActivityStamp::default(),
                activity: super::super::ResidentTextureActivity::enabled(),
            },
        );
        Ok(texture)
    }

    pub(crate) fn ensure_mipmapped_render_target_texture(
        &mut self,
        device: &Device,
        target: DecodedRenderTarget,
        discard_guest_contents: bool,
    ) -> Result<SelectedResidentTexture, String> {
        if target.texture_type != 9
            || target.layer_count != 1
            || target.mip_level_count <= 1
            || target.sample_count != 1
        {
            return Err(format!(
                "mipmapped render target requires a single-sample 2D resource, got type={} layers={} levels={} samples={}",
                target.texture_type,
                target.layer_count,
                target.mip_level_count,
                target.sample_count
            ));
        }
        if !discard_guest_contents && self.rematerialise_stale_resident_for_read(target.texture)? {
            return Ok(SelectedResidentTexture {
                texture: self
                    .resident_guest_textures
                    .get(&target.texture)
                    .expect("rematerialised mip target exists")
                    .texture
                    .clone(),
                descriptor: target.texture,
            });
        }
        if let Some(resident) = self
            .resident_guest_textures
            .get_mut(&target.texture)
            .filter(|resident| {
                resident.texture_type == 9
                    && resident.layer_count == 1
                    && resident.mip_level_count >= target.mip_level_count
            })
        {
            match refresh_resident_texture_write_watch(resident)? {
                WriteWatchRefresh::Active
                | WriteWatchRefresh::Reprotecting
                | WriteWatchRefresh::Renewed { .. } => {
                    if resident.reclaim_for_bind(discard_guest_contents) {
                        return Ok(SelectedResidentTexture {
                            texture: resident.texture.clone(),
                            descriptor: target.texture,
                        });
                    }
                }
                WriteWatchRefresh::Replaced => {}
            }
        }
        if self
            .resident_guest_textures
            .get(&target.texture)
            .is_some_and(|resident| resident.gpu_written())
        {
            self.ensure_active_open_guest_batch()?;
            let command_buffer = {
                let batch = self
                    .active_open_guest_batch_mut()
                    .expect("guest batch was opened for mip promotion");
                batch.close_active_pass_encoders();
                batch.command_buffer.clone()
            };
            retire_overlapping_sampled_textures(&mut self.sampled_texture_cache, target.span);
            let resident = self
                .resident_guest_textures
                .remove(&target.texture)
                .expect("selected GPU-written target exists");
            let descriptor = DecodedImageDescriptor {
                guest: target.texture,
                span: target.span,
                destination_select: 0,
                base_mip_level: 0,
                mip_level_count: target.mip_level_count,
                texture_type: target.texture_type,
                layer_count: target.layer_count,
            };
            let promoted = promote_resident_mip_hierarchy(
                device,
                &command_buffer,
                descriptor,
                resident,
                Mode27WorkOrigin::RenderTargetCreate,
            )?;
            let texture = promoted.texture.clone();
            self.resident_guest_textures
                .insert(target.texture, promoted);
            self.active_open_guest_batch_mut()
                .expect("mip-promotion batch remains open")
                .record_resident_write_footprint(
                    SharedEventWriteKind::InternalResidentDestination,
                    target.span,
                );
            self.record_internal_resident_version(target.texture);
            return Ok(SelectedResidentTexture {
                texture,
                descriptor: target.texture,
            });
        }
        if let Some(resident) = self.resident_guest_textures.remove(&target.texture) {
            trace_resident_texture_removal("mipmapped-target-resident-replaced", &resident);
        }
        retire_overlapping_sampled_textures(&mut self.sampled_texture_cache, target.span);
        let started = Instant::now();
        let layout = mode27_mip_chain_layout(target.texture, target.mip_level_count)?;
        let write_watch = watch_resident_texture(
            target.texture,
            target.texture_type,
            target.layer_count,
            target.mip_level_count,
        )?;
        let raw =
            snapshot_watched_guest_range(target.texture.address, layout.byte_count, &write_watch)?;
        let (_, linear_levels) =
            decode_mode27_mip_chain_from_raw(target.texture, target.mip_level_count, &raw)?;
        let texture = create_mipmapped_guest_texture(
            device,
            target.texture,
            target.mip_level_count,
            &linear_levels,
        )?;
        record_mode27_work_detail(
            Mode27WorkOrigin::RenderTargetCreate,
            target.texture,
            target.mip_level_count,
            layout.byte_count,
            started.elapsed(),
        );
        texture.set_label(&format!(
            "REMu {}-level guest image {:#x}",
            target.mip_level_count, target.texture.address
        ));
        let (span, owned) = ResidentGuestTexture::declared_authority(&write_watch);
        self.resident_guest_textures.insert(
            target.texture,
            ResidentGuestTexture {
                texture: texture.clone(),
                component_reorder_view: None,
                descriptor: target.texture,
                texture_type: target.texture_type,
                layer_count: target.layer_count,
                mip_level_count: target.mip_level_count,
                write_watch,
                span,
                owned,
                gpu_authored: RangeSet::default(),
                gpu_content_executed: false,
                state: ResidencyState::Live,
                acquire_invalidated: false,
                content_version: 0,
                gpu_write_watch_generations: Vec::new(),
                last_use: ResidentTextureActivityStamp::default(),
                activity: super::super::ResidentTextureActivity::enabled(),
            },
        );
        Ok(SelectedResidentTexture {
            texture,
            descriptor: target.texture,
        })
    }

    pub(crate) fn ensure_layered_render_target_texture(
        &mut self,
        device: &Device,
        target: DecodedRenderTarget,
        discard_guest_contents: bool,
    ) -> Result<SelectedResidentTexture, String> {
        if target.texture_type == 9 && target.layer_count == 1 {
            return if target.mip_level_count == 1 {
                self.ensure_resident_texture_with_initial_contents(
                    device,
                    target.texture,
                    target.span,
                    !discard_guest_contents,
                )
            } else {
                self.ensure_mipmapped_render_target_texture(device, target, discard_guest_contents)
            };
        }
        if target.texture_type != 10 || target.layer_count <= 1 {
            return Err(format!(
                "unsupported layered target texture type={} layers={}",
                target.texture_type, target.layer_count
            ));
        }
        if target.sample_count != 1 {
            return Err("multisampled 3D color targets are not implemented".to_owned());
        }
        if target.mip_level_count != 1 || target.mip_level != 0 {
            return Err(format!(
                "mipmapped 3D color targets are not implemented (mip={}/{})",
                target.mip_level,
                target.mip_level_count - 1
            ));
        }
        if !discard_guest_contents && self.rematerialise_stale_resident_for_read(target.texture)? {
            return Ok(SelectedResidentTexture {
                texture: self
                    .resident_guest_textures
                    .get(&target.texture)
                    .expect("rematerialised layered target exists")
                    .texture
                    .clone(),
                descriptor: target.texture,
            });
        }
        if let Some(resident) = self
            .resident_guest_textures
            .get_mut(&target.texture)
            .filter(|resident| {
                resident.texture_type == target.texture_type
                    && resident.layer_count == target.layer_count
                    && resident.mip_level_count == 1
            })
        {
            match refresh_resident_texture_write_watch(resident)? {
                WriteWatchRefresh::Active
                | WriteWatchRefresh::Reprotecting
                | WriteWatchRefresh::Renewed { .. } => {
                    if resident.reclaim_for_bind(discard_guest_contents) {
                        return Ok(SelectedResidentTexture {
                            texture: resident.texture.clone(),
                            descriptor: target.texture,
                        });
                    }
                }
                WriteWatchRefresh::Replaced => {}
            }
        }
        if self
            .resident_guest_textures
            .get(&target.texture)
            .is_some_and(|resident| resident.gpu_written())
        {
            return Err(format!(
                "cannot replace GPU-written target {:#x} with a {}-layer 3D view",
                target.texture.address, target.layer_count
            ));
        }
        if let Some(resident) = self.resident_guest_textures.remove(&target.texture) {
            trace_resident_texture_removal("layered-target-resident-replaced", &resident);
        }
        retire_overlapping_sampled_textures(&mut self.sampled_texture_cache, target.span);
        let descriptor = DecodedImageDescriptor {
            guest: target.texture,
            span: target.span,
            destination_select: 0,
            base_mip_level: 0,
            mip_level_count: 1,
            texture_type: target.texture_type,
            layer_count: target.layer_count,
        };
        let write_watch = watch_sampled_texture(descriptor)?;
        let raw = snapshot_watched_sampled_texture_raw(descriptor, &write_watch)?;
        let linear = decode_sampled_texture_linear_from_raw(descriptor, &raw)?;
        let texture = create_layered_render_target_texture(device, descriptor, &linear)?;
        texture.set_label(&format!(
            "REMu {}-layer guest volume {:#x}",
            target.layer_count, target.texture.address
        ));
        let (span, owned) = ResidentGuestTexture::declared_authority(&write_watch);
        self.resident_guest_textures.insert(
            target.texture,
            ResidentGuestTexture {
                texture: texture.clone(),
                component_reorder_view: None,
                descriptor: target.texture,
                texture_type: target.texture_type,
                layer_count: target.layer_count,
                mip_level_count: 1,
                write_watch,
                span,
                owned,
                gpu_authored: RangeSet::default(),
                gpu_content_executed: false,
                state: ResidencyState::Live,
                acquire_invalidated: false,
                content_version: 0,
                gpu_write_watch_generations: Vec::new(),
                last_use: ResidentTextureActivityStamp::default(),
                activity: super::super::ResidentTextureActivity::enabled(),
            },
        );
        Ok(SelectedResidentTexture {
            texture,
            descriptor: target.texture,
        })
    }

    pub(crate) fn ensure_depth_render_target_texture(
        &mut self,
        device: &Device,
        depth: DecodedDepthTarget,
        minimum_width: u32,
        minimum_height: u32,
    ) -> Result<SelectedResidentTexture, String> {
        let selected = if depth.layer_count == 1 {
            self.ensure_resident_texture_with_initial_contents(
                device,
                depth.texture,
                depth.span,
                true,
            )?
        } else {
            self.ensure_layered_depth_render_target_texture(device, depth)?
        };
        let host_width = depth.texture.width.max(minimum_width);
        let host_height = depth.texture.height.max(minimum_height);
        if selected.texture.width() >= u64::from(host_width)
            && selected.texture.height() >= u64::from(host_height)
        {
            return Ok(selected);
        }

        // Unreal can bind a hierarchically sized DB surface beside a larger color target. The
        // console expands the fast-cleared depth allocation to the active framebuffer extent;
        // Metal instead requires every attachment to cover the explicit render-pass extent.
        // Promote the host depth image, initialize the extra area to a comparison-neutral value,
        // and preserve the guest-declared portion. This prevents menus and scenes from being
        // clipped to the small DB metadata extent.
        self.wait_for_guest_command_batches()?;
        let queue = self
            .command_queue
            .as_ref()
            .ok_or("Metal command queue is unavailable")?
            .clone();
        let metal_descriptor = TextureDescriptor::new();
        metal_descriptor.set_texture_type(if depth.layer_count == 1 {
            MTLTextureType::D2
        } else {
            MTLTextureType::D2Array
        });
        metal_descriptor.set_pixel_format(depth.texture.pixel_format);
        metal_descriptor.set_width(u64::from(host_width));
        metal_descriptor.set_height(u64::from(host_height));
        if depth.layer_count > 1 {
            metal_descriptor.set_array_length(u64::from(depth.layer_count));
        }
        metal_descriptor.set_storage_mode(MTLStorageMode::Shared);
        metal_descriptor.set_usage(
            MTLTextureUsage::RenderTarget
                | MTLTextureUsage::ShaderRead
                | MTLTextureUsage::ShaderWrite,
        );
        let promoted = device.new_texture(&metal_descriptor);
        promoted.set_label(&format!(
            "REMu promoted {}-layer depth image {:#x} {}x{}",
            depth.layer_count, depth.texture.address, host_width, host_height
        ));

        let command_buffer = queue.new_command_buffer().to_owned();
        command_buffer.set_label("REMu depth-target promotion");
        let pass = RenderPassDescriptor::new();
        pass.set_render_target_width(u64::from(host_width));
        pass.set_render_target_height(u64::from(host_height));
        pass.set_render_target_array_length(u64::from(depth.layer_count));
        let attachment = pass
            .depth_attachment()
            .ok_or("Metal render pass has no depth attachment")?;
        attachment.set_texture(Some(&promoted));
        if depth.layer_count > 1 {
            attachment.set_slice(0);
        }
        attachment.set_load_action(MTLLoadAction::Clear);
        attachment.set_store_action(MTLStoreAction::Store);
        attachment.set_clear_depth(initial_depth_clear_for_target(depth));
        let encoder = command_buffer.new_render_command_encoder(pass);
        encoder.end_encoding();

        let blit = command_buffer.new_blit_command_encoder();
        for layer in 0..depth.layer_count {
            blit.copy_from_texture(
                &selected.texture,
                u64::from(layer),
                0,
                MTLOrigin { x: 0, y: 0, z: 0 },
                MTLSize::new(
                    selected.texture.width().min(u64::from(host_width)),
                    selected.texture.height().min(u64::from(host_height)),
                    1,
                ),
                &promoted,
                u64::from(layer),
                0,
                MTLOrigin { x: 0, y: 0, z: 0 },
            );
        }
        blit.end_encoding();
        commit_and_settle(&command_buffer, "depth-target promotion")?;

        let resident = self
            .resident_guest_textures
            .get_mut(&selected.descriptor)
            .ok_or("promoted depth target lost its resident identity")?;
        resident.texture = promoted.clone();
        resident.component_reorder_view = None;
        Ok(SelectedResidentTexture {
            texture: promoted,
            descriptor: selected.descriptor,
        })
    }

    fn ensure_layered_depth_render_target_texture(
        &mut self,
        device: &Device,
        depth: DecodedDepthTarget,
    ) -> Result<SelectedResidentTexture, String> {
        if depth.layer_count <= 1 || depth.sample_count != 1 {
            return Err(format!(
                "layered depth target requires multiple single-sample slices, got layers={} samples={}",
                depth.layer_count, depth.sample_count
            ));
        }
        const DEPTH_ARRAY_TEXTURE_TYPE: u32 = 13;
        if self.rematerialise_stale_resident_for_read(depth.texture)? {
            return Ok(SelectedResidentTexture {
                texture: self
                    .resident_guest_textures
                    .get(&depth.texture)
                    .expect("rematerialised depth array exists")
                    .texture
                    .clone(),
                descriptor: depth.texture,
            });
        }
        if let Some(resident) =
            self.resident_guest_textures
                .get_mut(&depth.texture)
                .filter(|resident| {
                    resident.texture_type == DEPTH_ARRAY_TEXTURE_TYPE
                        && resident.layer_count == depth.layer_count
                        && resident.mip_level_count == 1
                })
        {
            match refresh_resident_texture_write_watch(resident)? {
                WriteWatchRefresh::Active
                | WriteWatchRefresh::Reprotecting
                | WriteWatchRefresh::Renewed { .. } => {
                    // Depth is tested as well as written, so a superseded array must rebuild.
                    if resident.reclaim_for_bind(false) {
                        return Ok(SelectedResidentTexture {
                            texture: resident.texture.clone(),
                            descriptor: depth.texture,
                        });
                    }
                }
                WriteWatchRefresh::Replaced => {}
            }
        }

        let face_bytes = guest_texture_byte_count(depth.texture)
            .ok_or("layered depth target face byte count overflows")?;
        let array_bytes = face_bytes
            .checked_mul(u64::from(depth.layer_count))
            .ok_or("layered depth target byte count overflows")?;
        let aliases = self
            .resident_guest_textures
            .values()
            .filter(|resident| {
                resident.gpu_written()
                    && resident
                        .write_watch
                        .overlaps_alias_range(depth.texture.address, array_bytes)
            })
            .map(|resident| resident.descriptor)
            .collect::<Vec<_>>();
        if !aliases.is_empty() {
            self.wait_for_guest_command_batches()?;
            for alias in aliases {
                let refresh = refresh_resident_texture_write_watch(
                    self.resident_guest_textures
                        .get_mut(&alias)
                        .expect("collected layered-depth alias exists"),
                )?;
                let cpu_reused = refresh != WriteWatchRefresh::Replaced
                    && self
                        .resident_guest_textures
                        .get_mut(&alias)
                        .expect("refreshed layered-depth alias exists")
                        .cpu_write_obsoletes_gpu_writeback(refresh)?;
                if refresh != WriteWatchRefresh::Replaced && !cpu_reused {
                    self.settle_resident_texture_writeback(
                        alias,
                        WritebackTrigger::LayeredDepthAlias,
                    )?;
                }
                let Some(resident) = self.resident_guest_textures.remove(&alias) else {
                    continue;
                };
                if refresh == WriteWatchRefresh::Replaced {
                    trace_resident_texture_removal(
                        "layered-depth-alias-mapping-replaced",
                        &resident,
                    );
                } else if cpu_reused {
                    trace_resident_texture_removal("cpu-reused-layered-depth-alias", &resident);
                }
            }
        }
        if let Some(resident) = self.resident_guest_textures.remove(&depth.texture) {
            trace_resident_texture_removal("layered-depth-resident-replaced", &resident);
        }

        let descriptor = DecodedImageDescriptor {
            guest: depth.texture,
            span: depth.span,
            destination_select: 0,
            base_mip_level: 0,
            mip_level_count: 1,
            texture_type: DEPTH_ARRAY_TEXTURE_TYPE,
            layer_count: depth.layer_count,
        };
        let write_watch = watch_sampled_texture(descriptor)?;
        retire_sampled_textures_overlapping_writeback(
            &mut self.sampled_texture_cache,
            &write_watch.alias_ranges(),
        );
        let raw = snapshot_watched_sampled_texture_raw(descriptor, &write_watch)?;
        let linear = decode_sampled_texture_linear_from_raw(descriptor, &raw)?;
        let texture = create_layered_depth_render_target_texture(device, descriptor, &linear)?;
        texture.set_label(&format!(
            "REMu {}-layer guest depth image {:#x}",
            depth.layer_count, depth.texture.address
        ));
        let (span, owned) = ResidentGuestTexture::declared_authority(&write_watch);
        self.resident_guest_textures.insert(
            depth.texture,
            ResidentGuestTexture {
                texture: texture.clone(),
                component_reorder_view: None,
                descriptor: depth.texture,
                texture_type: DEPTH_ARRAY_TEXTURE_TYPE,
                layer_count: depth.layer_count,
                mip_level_count: 1,
                write_watch,
                span,
                owned,
                gpu_authored: RangeSet::default(),
                gpu_content_executed: false,
                state: ResidencyState::Live,
                acquire_invalidated: false,
                content_version: 0,
                gpu_write_watch_generations: Vec::new(),
                last_use: ResidentTextureActivityStamp::default(),
                activity: super::super::ResidentTextureActivity::enabled(),
            },
        );
        Ok(SelectedResidentTexture {
            texture,
            descriptor: depth.texture,
        })
    }

    pub(crate) fn ensure_direct_scanout_texture(
        &mut self,
        device: &Device,
        descriptor: GuestTextureDescriptor,
    ) -> Texture {
        if let Some(scanout) = self.direct_scanout_textures.get(&descriptor.address)
            && scanout.descriptor == descriptor
        {
            return scanout.texture.clone();
        }
        let metal_descriptor = TextureDescriptor::new();
        metal_descriptor.set_texture_type(MTLTextureType::D2);
        metal_descriptor.set_pixel_format(descriptor.pixel_format);
        metal_descriptor.set_width(u64::from(descriptor.width));
        metal_descriptor.set_height(u64::from(descriptor.height));
        metal_descriptor.set_storage_mode(MTLStorageMode::Private);
        metal_descriptor.set_usage(MTLTextureUsage::RenderTarget | MTLTextureUsage::ShaderRead);
        let texture = device.new_texture(&metal_descriptor);
        self.direct_scanout_textures.insert(
            descriptor.address,
            DirectScanoutTexture {
                descriptor,
                texture: texture.clone(),
                initialized: false,
            },
        );
        texture
    }

    pub(crate) fn ensure_scanout_snapshot_texture(
        &mut self,
        device: &Device,
        descriptor: GuestTextureDescriptor,
    ) -> Texture {
        if let Some(snapshot) = self.scanout_snapshot_textures.get(&descriptor.address)
            && snapshot.descriptor == descriptor
        {
            return snapshot.texture.clone();
        }
        let metal_descriptor = TextureDescriptor::new();
        metal_descriptor.set_texture_type(MTLTextureType::D2);
        metal_descriptor.set_pixel_format(descriptor.pixel_format);
        metal_descriptor.set_width(u64::from(descriptor.width));
        metal_descriptor.set_height(u64::from(descriptor.height));
        metal_descriptor.set_storage_mode(MTLStorageMode::Private);
        metal_descriptor.set_usage(MTLTextureUsage::ShaderRead);
        let texture = device.new_texture(&metal_descriptor);
        texture.set_label(&format!("REMu scanout snapshot {:#x}", descriptor.address));
        self.scanout_snapshot_textures.insert(
            descriptor.address,
            ScanoutSnapshotTexture {
                descriptor,
                texture: texture.clone(),
            },
        );
        texture
    }

    pub(crate) fn ensure_multisample_texture(
        &mut self,
        device: &Device,
        descriptor: GuestTextureDescriptor,
        sample_count: u32,
        minimum_width: u32,
        minimum_height: u32,
    ) -> Result<(Texture, bool), String> {
        if sample_count <= 1 {
            return Err("multisample texture requires at least two samples".to_owned());
        }
        if !device.supports_texture_sample_count(u64::from(sample_count)) {
            return Err(format!(
                "Metal device does not support {sample_count} samples for {:?}",
                descriptor.pixel_format
            ));
        }
        let key = MultisampleTextureKey {
            descriptor,
            sample_count,
        };
        let host_width = descriptor.width.max(minimum_width);
        let host_height = descriptor.height.max(minimum_height);
        if let Some(resident) = self.multisample_textures.get(&key)
            && resident.texture.width() >= u64::from(host_width)
            && resident.texture.height() >= u64::from(host_height)
        {
            return Ok((resident.texture.clone(), !resident.initialized));
        }
        self.multisample_textures.remove(&key);
        let metal_descriptor = TextureDescriptor::new();
        metal_descriptor.set_texture_type(MTLTextureType::D2Multisample);
        metal_descriptor.set_pixel_format(descriptor.pixel_format);
        metal_descriptor.set_width(u64::from(host_width));
        metal_descriptor.set_height(u64::from(host_height));
        metal_descriptor.set_sample_count(u64::from(sample_count));
        metal_descriptor.set_storage_mode(MTLStorageMode::Private);
        metal_descriptor.set_usage(MTLTextureUsage::RenderTarget);
        let texture = device.new_texture(&metal_descriptor);
        texture.set_label(&format!(
            "REMu {}x multisample guest image {:#x}",
            sample_count, descriptor.address
        ));
        self.multisample_textures.insert(
            key,
            MultisampleResidentTexture {
                texture: texture.clone(),
                initialized: false,
            },
        );
        Ok((texture, true))
    }

    pub(crate) fn mark_multisample_texture_initialized(
        &mut self,
        descriptor: GuestTextureDescriptor,
        sample_count: u32,
    ) {
        if let Some(resident) = self.multisample_textures.get_mut(&MultisampleTextureKey {
            descriptor,
            sample_count,
        }) {
            resident.initialized = true;
        }
    }

    #[cfg(test)]
    pub(crate) fn mark_resident_texture_written(&mut self, descriptor: GuestTextureDescriptor) {
        self.mark_resident_texture_written_at(
            descriptor,
            ResourceSpan::for_texture(descriptor),
            None,
            WrittenRegion::whole_resource(),
            ResidentTextureActivityStamp::default(),
        );
    }

    /// Records a GPU write against `descriptor`'s resident and retires its stale interpretations.
    ///
    /// `span` is `descriptor`'s span as the decode path resolved it. `selected_resident` is the
    /// residency key the ensure path actually served this write's texture from. Only attribution
    /// follows it; sampled and multisample retirement stay keyed on `descriptor`, because the
    /// written bytes are the ones the guest descriptor names. `region` bounds the write inside the
    /// resident it is attributed to.
    pub(crate) fn mark_resident_texture_written_at(
        &mut self,
        descriptor: GuestTextureDescriptor,
        span: Option<ResourceSpan>,
        selected_resident: Option<GuestTextureDescriptor>,
        region: WrittenRegion,
        activity_stamp: ResidentTextureActivityStamp,
    ) {
        self.mark_resident_texture_written_preserving_multisample_at(
            descriptor,
            span,
            selected_resident,
            region,
            activity_stamp,
        );
        self.multisample_textures
            .retain(|key, _| !spans_overlap(ResourceSpan::for_texture(key.descriptor), span));
    }

    pub(crate) fn mark_resident_texture_written_preserving_multisample_at(
        &mut self,
        descriptor: GuestTextureDescriptor,
        span: Option<ResourceSpan>,
        selected_resident: Option<GuestTextureDescriptor>,
        region: WrittenRegion,
        activity_stamp: ResidentTextureActivityStamp,
    ) {
        self.resident_content_version = self.resident_content_version.saturating_add(1);
        let version = self.resident_content_version;
        // The first two arms name the exact allocation the write landed in, whatever state it is
        // in: recording the write is what makes it the authority again. Only the scan may not
        // reach a demoted resident, which answers for no byte and must not be handed any.
        let resident_descriptor = selected_resident
            .filter(|selected| self.resident_guest_textures.contains_key(selected))
            .or_else(|| {
                self.resident_guest_textures
                    .contains_key(&descriptor)
                    .then_some(descriptor)
            })
            .or_else(|| {
                self.resident_guest_textures
                    .values()
                    .find(|resident| {
                        resident.is_live()
                            && guest_texture_descriptors_share_storage(
                                resident.descriptor,
                                descriptor,
                            )
                    })
                    .map(|resident| resident.descriptor)
            });
        if resident_descriptor.is_none()
            && self.resident_guest_textures.values().any(|resident| {
                resident.is_live()
                    && guest_texture_descriptors_share_gpu_sample_storage(
                        resident.descriptor,
                        descriptor,
                    )
            })
        {
            // A resident that only the pitch-tolerant rule can reach served this write, and the
            // caller did not carry its key here. Ownership and content version are now lost.
            RESIDENT_WRITE_ATTRIBUTION_MISSES.fetch_add(1, Ordering::Relaxed);
        }
        let mut footprint_span = None;
        let mut texture_version = None;
        if let Some(writer) = resident_descriptor {
            let (written, writer_span) = {
                let resident = self
                    .resident_guest_textures
                    .get(&writer)
                    .expect("attributed writer exists");
                match resident.span {
                    Some(writer_span) if range_algebra_enabled() => (
                        written_byte_ranges(
                            writer_span,
                            resident.descriptor,
                            resident.texture_type,
                            resident.layer_count,
                            resident.mip_level_count,
                            region,
                        ),
                        Some(writer_span),
                    ),
                    // No physical identity — or the kill switch off — claims the whole declared
                    // range, which supersedes every overlapping resident exactly as the
                    // descriptor-level retirement did.
                    resident_span => (
                        resident_span
                            .as_ref()
                            .map(RangeSet::for_span)
                            .unwrap_or_default(),
                        resident_span,
                    ),
                }
            };
            footprint_span = writer_span;
            {
                let resident = self
                    .resident_guest_textures
                    .get_mut(&writer)
                    .expect("attributed writer exists");
                resident.note_write(activity_stamp);
                resident.note_gpu_authored(&written);
                debug_assert!(
                    resident.gpu_written(),
                    "a GPU write must leave its resident owing the bytes it covered"
                );
                resident.content_version = version;
                resident.note_gpu_write();
                if let Some(writer_span) = writer_span {
                    texture_version = Some(SharedEventResidentTextureVersion {
                        texture: resident.texture.clone(),
                        descriptor: resident.descriptor,
                        texture_type: resident.texture_type,
                        layer_count: resident.layer_count,
                        mip_level_count: resident.mip_level_count,
                        content_version: resident.content_version,
                        span: writer_span,
                        kind: SharedEventWriteKind::ResidentDestination,
                    });
                }
            }
            if let Some(writer_span) = writer_span {
                self.subtract_resident_authority(
                    &writer_span,
                    &written,
                    Some(writer),
                    StaleSource::Allocation {
                        descriptor: writer,
                        content_version: version,
                    },
                    "gpu-write-supersedes-alias",
                );
            }
        }
        if let Some(batch) = self.active_open_guest_batch_mut() {
            batch.record_resident_write_footprint(
                SharedEventWriteKind::ResidentDestination,
                footprint_span,
            );
            if let Some(texture_version) = texture_version {
                batch.record_resident_texture_version(texture_version);
            }
        }
        retire_overlapping_sampled_textures(&mut self.sampled_texture_cache, span);
    }

    /// Guest RAM became authoritative for `published` of `span`'s storage.
    ///
    /// `writer` keeps its own authority: RAM merely agrees with it now. Everyone else loses those
    /// bytes, and a resident that loses its last one is demoted rather than destroyed, so its
    /// allocation, watch and fingerprint baseline survive to make the rebuild a validated no-op.
    pub(crate) fn note_guest_memory_authoritative(
        &mut self,
        span: &ResourceSpan,
        published: &RangeSet,
        writer: Option<GuestTextureDescriptor>,
        reason: &str,
    ) {
        self.subtract_resident_authority(span, published, writer, StaleSource::GuestMemory, reason);
    }

    /// Hands `writer`'s authority to guest RAM after its contents were published there.
    pub(crate) fn note_resident_writeback_published(&mut self, writer: GuestTextureDescriptor) {
        let Some((span, published)) = self
            .resident_guest_textures
            .get(&writer)
            .and_then(|resident| Some((resident.span?, resident.owned.clone())))
        else {
            return;
        };
        self.note_guest_memory_authoritative(
            &span,
            &published,
            Some(writer),
            "superseded-by-writeback",
        );
    }

    /// Takes `taken` away from every resident but `keep` whose span the index query reaches.
    ///
    /// Candidates keep the operand order the linear scans they replace used. Only residents whose
    /// runs live in the same offset space are subtracted byte for byte; an identity that cannot be
    /// compared is superseded whole, which is what the descriptor-overlap rule did to every one.
    fn subtract_resident_authority(
        &mut self,
        span: &ResourceSpan,
        taken: &RangeSet,
        keep: Option<GuestTextureDescriptor>,
        source: StaleSource,
        reason: &str,
    ) {
        let candidates = self
            .resident_guest_textures
            .overlapping(span, SpanQueryOrder::ResidentFirst);
        for candidate in candidates {
            if Some(candidate) == keep {
                continue;
            }
            let Some(other) = self.resident_guest_textures.get(&candidate) else {
                continue;
            };
            if !other.is_live() {
                // An already demoted allocation holds nothing left to take.
                continue;
            }
            // An R16/R32 depth-sample SRD at the DB address must not demote the native Depth*
            // resident. Silent Hill title: post-prepass samples bound an empty gpu-authored
            // R32Float shadow (magnitude 0) while Depth32Float still held Z (gpufalse after
            // subtract) → depth_native=0 and lighting/tonemap read clear depth.
            if let Some(writer) = keep
                && depth_sample_view_must_not_steal_depth_authority(writer, candidate)
            {
                OWNED_SUBTRACT_TENANTS_SURVIVED.fetch_add(1, Ordering::Relaxed);
                continue;
            }
            // ResourceSpan::overlaps consults shared-backing identity and can report neighbours
            // in the same heap that do not share virtual bytes (Silent Hill title intermediate
            // @ 0x8044590000 vs RGBA16Float @ 0x8043490000). Never steal authority from a
            // different-address resident unless the descriptors actually share storage or
            // overlap in guest VA.
            if let Some(writer) = keep.filter(|writer| writer.address != candidate.address) {
                let shares = guest_texture_descriptors_share_byte_storage(writer, candidate);
                let overlaps_va = guest_texture_byte_count(writer).is_some_and(|left_bytes| {
                    guest_texture_byte_count(candidate).is_some_and(|right_bytes| {
                        guest_ranges_overlap(
                            writer.address,
                            left_bytes,
                            candidate.address,
                            right_bytes,
                        )
                    })
                });
                if !shares && !overlaps_va {
                    OWNED_SUBTRACT_TENANTS_SURVIVED.fetch_add(1, Ordering::Relaxed);
                    continue;
                }
            }
            let comparable = other
                .span
                .is_some_and(|other_span| other_span.shares_offset_space(span));
            if comparable && !other.owned.intersects(taken) {
                // A disjoint tenant of the same pooled allocation keeps every byte it holds.
                OWNED_SUBTRACT_TENANTS_SURVIVED.fetch_add(1, Ordering::Relaxed);
                continue;
            }
            if other.write_watch.retirement() == Some(CpuWriteWatchRetirement::Replaced) {
                // A replaced mapping proves nothing about any byte, so it cannot be demoted.
                if let Some(resident) = self.resident_guest_textures.remove(&candidate) {
                    trace_resident_texture_removal("superseded-mapping-replaced", &resident);
                }
                continue;
            }
            let other = self
                .resident_guest_textures
                .get_mut(&candidate)
                .expect("inspected candidate is still resident");
            if comparable {
                other.subtract_authority(taken, source);
            } else {
                other.demote(source);
            }
            if !other.is_live() {
                trace_resident_texture_demotion(reason, other);
            }
        }
    }

    pub(crate) fn note_resident_texture_read_at(
        &mut self,
        descriptor: GuestTextureDescriptor,
        activity_stamp: ResidentTextureActivityStamp,
    ) {
        let resident_descriptor = self
            .resident_guest_textures
            .contains_key(&descriptor)
            .then_some(descriptor)
            .or_else(|| {
                self.resident_guest_textures
                    .values()
                    .find(|resident| {
                        resident.is_live()
                            && guest_texture_descriptors_share_storage(
                                resident.descriptor,
                                descriptor,
                            )
                    })
                    .map(|resident| resident.descriptor)
            });
        if let Some(resident) = resident_descriptor.and_then(|resident_descriptor| {
            self.resident_guest_textures.get_mut(&resident_descriptor)
        }) {
            resident.note_read(activity_stamp);
        }
    }
}

pub(crate) fn ensure_resident_texture_in_map(
    device: &Device,
    queue: Option<&metal::CommandQueue>,
    descriptor: GuestTextureDescriptor,
    resident_guest_textures: &mut ResidentTextureMap,
    upload_guest_contents: bool,
) -> Result<Texture, String> {
    if let Some(resident) = resident_guest_textures
        .get_mut(&descriptor)
        .filter(|resident| {
            resident.texture_type == 9 && resident.layer_count == 1 && resident.mip_level_count >= 1
        })
    {
        match refresh_resident_texture_write_watch(resident)? {
            WriteWatchRefresh::Active
            | WriteWatchRefresh::Reprotecting
            | WriteWatchRefresh::Renewed { .. } => {
                if resident.reclaim_for_bind(!upload_guest_contents) {
                    return Ok(resident.texture.clone());
                }
            }
            WriteWatchRefresh::Replaced => {}
        }
    }
    if let Some(resident) = resident_guest_textures.remove(&descriptor) {
        let reason = if resident.write_watch.retirement() == Some(CpuWriteWatchRetirement::Replaced)
        {
            "ensure-map-mapping-replaced"
        } else if !resident.serves_reads() {
            // The superseding allocation is authoritative for the overlapping bytes and has not
            // published them yet, so this rebuild reads whatever guest RAM holds. That is the
            // same answer the descriptor-level retirement gave; PR3 re-materialises on the GPU.
            "ensure-map-superseded-rebuild"
        } else {
            "ensure-map-incompatible-resident"
        };
        trace_resident_texture_removal(reason, &resident);
    }
    let write_watch = watch_guest_texture(descriptor)?;
    let usage = MTLTextureUsage::RenderTarget
        | MTLTextureUsage::ShaderRead
        | MTLTextureUsage::ShaderWrite
        | MTLTextureUsage::PixelFormatView;
    let texture = if upload_guest_contents {
        if !gpu_tiler_owns_descriptor(descriptor) {
            return Err(format!(
                "Phase 1G: refusing CPU snapshot upload for tile={} format={:?} bpe={} — no GPU tiler ownership",
                descriptor.tile_mode, descriptor.pixel_format, descriptor.bytes_per_pixel
            ));
        }
        let queue = queue.ok_or_else(|| {
            format!(
                "Phase 1G: ensure_resident_texture_in_map needs a Metal queue for GPU upload of tile={} at {:#x}",
                descriptor.tile_mode, descriptor.address
            )
        })?;
        let texture = create_uninitialized_guest_texture(device, descriptor, usage)?;
        let upload_buffer = queue.new_command_buffer().to_owned();
        upload_buffer.set_label("REMu Phase 1G ensure-map GPU upload");
        encode_gpu_upload_from_guest(device, &upload_buffer, descriptor, &texture)?;
        upload_buffer.commit();
        upload_buffer.wait_until_completed();
        texture
    } else {
        create_uninitialized_guest_texture(device, descriptor, usage)?
    };
    texture.set_label(&format!("REMu guest image {:#x}", descriptor.address));
    let (span, owned) = ResidentGuestTexture::declared_authority(&write_watch);
    resident_guest_textures.insert(
        descriptor,
        ResidentGuestTexture {
            texture: texture.clone(),
            component_reorder_view: None,
            descriptor,
            texture_type: 9,
            layer_count: 1,
            mip_level_count: 1,
            write_watch,
            span,
            owned,
            gpu_authored: RangeSet::default(),
                gpu_content_executed: false,
            state: ResidencyState::Live,
            acquire_invalidated: false,
            content_version: 0,
            gpu_write_watch_generations: Vec::new(),
            last_use: ResidentTextureActivityStamp::default(),
            activity: super::super::ResidentTextureActivity::enabled(),
        },
    );
    Ok(texture)
}

pub(crate) fn create_guest_texture(
    device: &Device,
    guest: GuestTextureDescriptor,
    usage: MTLTextureUsage,
    linear: &[u8],
) -> Result<Texture, String> {
    let (elements_wide, elements_high, _, bytes_per_element) = texture_element_layout(guest)?;
    let required = u64::try_from(elements_wide)
        .ok()
        .and_then(|width| width.checked_mul(u64::try_from(elements_high).ok()?))
        .and_then(|elements| elements.checked_mul(u64::try_from(bytes_per_element).ok()?))
        .and_then(|bytes| usize::try_from(bytes).ok())
        .ok_or("Metal texture dimensions overflow")?;
    if guest.width == 0 || guest.height == 0 || linear.len() != required {
        return Err(format!(
            "invalid guest texture upload {}x{} format {:?} with {} bytes (expected {required})",
            guest.width,
            guest.height,
            guest.pixel_format,
            linear.len()
        ));
    }
    let texture = create_uninitialized_guest_texture(device, guest, usage)?;
    texture.replace_region(
        MTLRegion::new_2d(0, 0, u64::from(guest.width), u64::from(guest.height)),
        0,
        linear.as_ptr().cast::<c_void>(),
        u64::try_from(elements_wide).expect("validated texture element width fits u64")
            * u64::try_from(bytes_per_element)
                .expect("validated texture element byte count fits u64"),
    );
    Ok(texture)
}

pub(crate) fn create_uninitialized_guest_texture(
    device: &Device,
    guest: GuestTextureDescriptor,
    usage: MTLTextureUsage,
) -> Result<Texture, String> {
    if guest.width == 0 || guest.height == 0 {
        return Err(format!(
            "invalid zero-sized guest texture {}x{} format {:?}",
            guest.width, guest.height, guest.pixel_format,
        ));
    }
    // Validate the guest/Metal format relationship even though a following render-pass clear,
    // rather than a CPU upload, supplies the initial contents.
    let _ = texture_element_layout(guest)?;
    let descriptor = TextureDescriptor::new();
    descriptor.set_texture_type(MTLTextureType::D2);
    descriptor.set_pixel_format(guest.pixel_format);
    descriptor.set_width(u64::from(guest.width));
    descriptor.set_height(u64::from(guest.height));
    descriptor.set_storage_mode(MTLStorageMode::Shared);
    descriptor.set_usage(usage);
    Ok(device.new_texture(&descriptor))
}

pub(crate) fn create_layered_render_target_texture(
    device: &Device,
    descriptor: DecodedImageDescriptor,
    linear: &[u8],
) -> Result<Texture, String> {
    if descriptor.texture_type != 10 || descriptor.layer_count <= 1 {
        return Err(format!(
            "layered render target requires a 3D texture, got type={} layers={}",
            descriptor.texture_type, descriptor.layer_count
        ));
    }
    let (elements_wide, elements_high, _, bytes_per_element) =
        texture_element_layout(descriptor.guest)?;
    let bytes_per_row = elements_wide
        .checked_mul(bytes_per_element)
        .ok_or("layered Metal texture row size overflows")?;
    let bytes_per_image = bytes_per_row
        .checked_mul(elements_high)
        .ok_or("layered Metal texture image size overflows")?;
    let required = bytes_per_image
        .checked_mul(
            usize::try_from(descriptor.layer_count)
                .map_err(|_| "layered Metal texture depth does not fit usize")?,
        )
        .ok_or("layered Metal texture byte count overflows")?;
    if linear.len() != required {
        return Err(format!(
            "invalid layered guest texture upload {}x{}x{} format {:?} with {} bytes (expected {required})",
            descriptor.guest.width,
            descriptor.guest.height,
            descriptor.layer_count,
            descriptor.guest.pixel_format,
            linear.len()
        ));
    }
    let metal_descriptor = TextureDescriptor::new();
    metal_descriptor.set_texture_type(MTLTextureType::D3);
    metal_descriptor.set_pixel_format(descriptor.guest.pixel_format);
    metal_descriptor.set_width(u64::from(descriptor.guest.width));
    metal_descriptor.set_height(u64::from(descriptor.guest.height));
    metal_descriptor.set_depth(u64::from(descriptor.layer_count));
    metal_descriptor.set_storage_mode(MTLStorageMode::Shared);
    metal_descriptor.set_usage(
        MTLTextureUsage::RenderTarget | MTLTextureUsage::ShaderRead | MTLTextureUsage::ShaderWrite,
    );
    let texture = device.new_texture(&metal_descriptor);
    texture.replace_region_in_slice(
        MTLRegion::new_3d(
            0,
            0,
            0,
            u64::from(descriptor.guest.width),
            u64::from(descriptor.guest.height),
            u64::from(descriptor.layer_count),
        ),
        0,
        0,
        linear.as_ptr().cast::<c_void>(),
        u64::try_from(bytes_per_row).expect("validated row size fits u64"),
        u64::try_from(bytes_per_image).expect("validated image size fits u64"),
    );
    Ok(texture)
}

fn create_layered_depth_render_target_texture(
    device: &Device,
    descriptor: DecodedImageDescriptor,
    linear: &[u8],
) -> Result<Texture, String> {
    if descriptor.texture_type != 13 || descriptor.layer_count <= 1 {
        return Err(format!(
            "layered depth target requires a 2D array, got type={} layers={}",
            descriptor.texture_type, descriptor.layer_count
        ));
    }
    let (elements_wide, elements_high, _, bytes_per_element) =
        texture_element_layout(descriptor.guest)?;
    let bytes_per_row = elements_wide
        .checked_mul(bytes_per_element)
        .ok_or("layered depth row size overflows")?;
    let bytes_per_image = bytes_per_row
        .checked_mul(elements_high)
        .ok_or("layered depth image size overflows")?;
    let required = bytes_per_image
        .checked_mul(
            usize::try_from(descriptor.layer_count)
                .map_err(|_| "layered depth layer count does not fit usize")?,
        )
        .ok_or("layered depth byte count overflows")?;
    if linear.len() != required {
        return Err(format!(
            "invalid layered depth upload {}x{}x{} format {:?} with {} bytes (expected {required})",
            descriptor.guest.width,
            descriptor.guest.height,
            descriptor.layer_count,
            descriptor.guest.pixel_format,
            linear.len()
        ));
    }
    let metal_descriptor = TextureDescriptor::new();
    metal_descriptor.set_texture_type(MTLTextureType::D2Array);
    metal_descriptor.set_pixel_format(descriptor.guest.pixel_format);
    metal_descriptor.set_width(u64::from(descriptor.guest.width));
    metal_descriptor.set_height(u64::from(descriptor.guest.height));
    metal_descriptor.set_array_length(u64::from(descriptor.layer_count));
    metal_descriptor.set_storage_mode(MTLStorageMode::Shared);
    metal_descriptor.set_usage(MTLTextureUsage::RenderTarget | MTLTextureUsage::ShaderRead);
    let texture = device.new_texture(&metal_descriptor);
    for layer in 0..descriptor.layer_count {
        let offset = usize::try_from(layer)
            .expect("validated depth layer fits usize")
            .checked_mul(bytes_per_image)
            .expect("validated depth layer offset fits usize");
        texture.replace_region_in_slice(
            MTLRegion::new_2d(
                0,
                0,
                u64::from(descriptor.guest.width),
                u64::from(descriptor.guest.height),
            ),
            0,
            u64::from(layer),
            linear[offset..].as_ptr().cast::<c_void>(),
            u64::try_from(bytes_per_row).expect("validated depth row size fits u64"),
            u64::try_from(bytes_per_image).expect("validated depth image size fits u64"),
        );
    }
    Ok(texture)
}
