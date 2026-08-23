// Copyright (C) 2026 REmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

//! Consumer-footprint classification for direct draws behind exact `SharedEvent` waits.
//!
//! Physical disjointness remains the default proof. Overlapping read-only globals can consume an
//! immutable copied-buffer version, and same-queue sampled images can consume an exact retained
//! Metal texture version. The consumer command buffer has already encoded the checkpoint's wait,
//! so neither path observes guest RAM or requires host retirement.

use super::super::{CB_COLOR_CONTROL, ColorOperationMode, ResourceSpan};
use super::super::{
    DEFAULT_CB_COLOR_CONTROL, GuestQueueBatchState, MetalPresenter, PhysicalFootprintRange,
    SharedEventDirectDrawPlan, SharedEventDrawTextureOverride, SharedEventFence,
    SharedEventProducerId, SharedEventSignalCheckpoint, SharedEventWriteKind,
    color_operation_is_raster_noop, decode_color_operation_mode,
    decode_depth_target_without_diagnostics, decode_image_descriptor_without_diagnostics,
    decode_render_targets, draw_diagnostics_are_host_effectful, effective_global_memory_binding,
    image_resource_descriptor_is_null, is_storage_image_opcode, is_writable_image_opcode,
    register_value, resident_texture_raw_byte_count, sampled_texture_raw_range,
};
use metal::foreign_types::ForeignType;
use remu_gpu::{IndexedDraw, ShaderSnapshot};
use remu_memory::GuestSharedMappingIdentity;
use remu_shader::{InstructionControl, ShaderEncoding, ShaderEvaluation, image_is_comparison};
use std::collections::HashMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

pub(crate) const SHARED_EVENT_DRAW_REJECT_COUNT: usize = 13;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(usize)]
pub(crate) enum SharedEventDrawReject {
    OverlapResident,
    OverlapScanout,
    OverlapCopiedGlobal,
    OverlapZeroCopyGlobal,
    OverlapTerminalWrite,
    OverlapInternalResident,
    IncompleteProducer,
    Unresolved,
    Indirect,
    ShaderCoverage,
    SpecialOperation,
    Diagnostic,
    Other,
}

impl SharedEventDrawReject {
    pub(crate) const fn index(self) -> usize {
        self as usize
    }

    const fn label(self) -> &'static str {
        match self {
            Self::OverlapResident => "overlap-resident",
            Self::OverlapScanout => "overlap-scanout",
            Self::OverlapCopiedGlobal => "overlap-copied-global",
            Self::OverlapZeroCopyGlobal => "overlap-zero-copy-global",
            Self::OverlapTerminalWrite => "overlap-terminal-write",
            Self::OverlapInternalResident => "overlap-internal-resident",
            Self::IncompleteProducer => "incomplete-producer",
            Self::Unresolved => "unresolved",
            Self::Indirect => "indirect",
            Self::ShaderCoverage => "shader-coverage",
            Self::SpecialOperation => "special-op",
            Self::Diagnostic => "diagnostic",
            Self::Other => "other",
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
#[repr(usize)]
enum DrawFootprintKind {
    ColorAttachment,
    DepthAttachment,
    GlobalRead,
    GlobalWrite,
    ScalarEvaluationRead,
    ImageRead,
    ImageWrite,
}

impl DrawFootprintKind {
    const COUNT: usize = 7;

    const fn index(self) -> usize {
        self as usize
    }

    const fn label(self) -> &'static str {
        match self {
            Self::ColorAttachment => "color",
            Self::DepthAttachment => "depth",
            Self::GlobalRead => "global-read",
            Self::GlobalWrite => "global-write",
            Self::ScalarEvaluationRead => "scalar-eval",
            Self::ImageRead => "image-read",
            Self::ImageWrite => "image-write",
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct DrawFootprintRange {
    kind: DrawFootprintKind,
    guest_address: u64,
    byte_count: u64,
    mapping: GuestSharedMappingIdentity,
    image: Option<super::super::DecodedImageDescriptor>,
}

impl DrawFootprintRange {
    fn new(
        kind: DrawFootprintKind,
        guest_address: u64,
        byte_count: u64,
    ) -> Result<Self, SharedEventDrawReject> {
        if byte_count == 0 || guest_address.checked_add(byte_count).is_none() {
            return Err(SharedEventDrawReject::Unresolved);
        }
        let span = ResourceSpan::new(guest_address, byte_count);
        let mapping = span.backing().ok_or(SharedEventDrawReject::Unresolved)?;
        mapping
            .backing_offset
            .checked_add(byte_count)
            .ok_or(SharedEventDrawReject::Unresolved)?;
        Ok(Self {
            kind,
            guest_address,
            byte_count,
            mapping,
            image: None,
        })
    }

    fn overlaps_producer(self, producer: PhysicalFootprintRange) -> bool {
        if self.mapping.backing_id != producer.mapping.backing_id {
            return false;
        }
        let Some(consumer_end) = self.mapping.backing_offset.checked_add(self.byte_count) else {
            return true;
        };
        let Some(producer_end) = producer
            .mapping
            .backing_offset
            .checked_add(producer.byte_count)
        else {
            return true;
        };
        self.mapping.backing_offset < producer_end && producer.mapping.backing_offset < consumer_end
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct DirectDrawFootprint {
    ranges: Arc<[DrawFootprintRange]>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum DirectDrawAdmission {
    Disjoint,
    VersionedGlobal,
    ResidentImage,
}

impl DirectDrawAdmission {
    const fn label(self) -> &'static str {
        match self {
            Self::Disjoint => "disjoint",
            Self::VersionedGlobal => "versioned-global",
            Self::ResidentImage => "resident-image",
        }
    }
}

impl DirectDrawFootprint {
    fn from_ranges(mut ranges: Vec<DrawFootprintRange>) -> Self {
        ranges.sort_unstable_by_key(|range| {
            (
                range.mapping.backing_id,
                range.kind,
                range.mapping.backing_offset,
                range.byte_count,
            )
        });
        let mut coalesced = Vec::<DrawFootprintRange>::with_capacity(ranges.len());
        for range in ranges {
            if let Some(previous) = coalesced.last_mut()
                && previous.kind == range.kind
                && previous.kind != DrawFootprintKind::GlobalRead
                && previous.kind != DrawFootprintKind::ImageRead
                && previous.mapping.backing_id == range.mapping.backing_id
            {
                let previous_end = previous
                    .mapping
                    .backing_offset
                    .checked_add(previous.byte_count)
                    .expect("validated draw footprint end");
                let range_end = range
                    .mapping
                    .backing_offset
                    .checked_add(range.byte_count)
                    .expect("validated draw footprint end");
                if range.mapping.backing_offset <= previous_end {
                    previous.byte_count =
                        previous_end.max(range_end) - previous.mapping.backing_offset;
                    continue;
                }
            }
            coalesced.push(range);
        }
        Self {
            ranges: coalesced.into(),
        }
    }

    fn byte_count(&self) -> u64 {
        self.ranges
            .iter()
            .fold(0_u64, |bytes, range| bytes.saturating_add(range.byte_count))
    }
}

fn append_range(
    ranges: &mut Vec<DrawFootprintRange>,
    kind: DrawFootprintKind,
    guest_address: u64,
    byte_count: u64,
) -> Result<(), SharedEventDrawReject> {
    ranges.push(DrawFootprintRange::new(kind, guest_address, byte_count)?);
    Ok(())
}

fn append_image_range(
    ranges: &mut Vec<DrawFootprintRange>,
    kind: DrawFootprintKind,
    descriptor: super::super::DecodedImageDescriptor,
    byte_count: u64,
) -> Result<(), SharedEventDrawReject> {
    let span = descriptor.span.ok_or(SharedEventDrawReject::Unresolved)?;
    if span.len() != byte_count {
        return Err(SharedEventDrawReject::Unresolved);
    }
    let mut range = DrawFootprintRange::new(kind, span.address(), byte_count)?;
    range.image = Some(descriptor);
    ranges.push(range);
    Ok(())
}

pub(crate) fn program_memory_is_completely_covered(
    snapshot: &ShaderSnapshot,
    evaluation: &ShaderEvaluation,
    allow_vertex_inputs: bool,
) -> bool {
    if !snapshot.supplemental_programs.is_empty() {
        return false;
    }
    snapshot.program.instructions.iter().all(|instruction| {
        if instruction.opcode.known_name().is_none() {
            return false;
        }
        match instruction.control {
            Some(InstructionControl::Image(_)) => evaluation
                .image_bindings
                .iter()
                .any(|binding| binding.pc == instruction.pc),
            Some(InstructionControl::GlobalMemory(_) | InstructionControl::ScalarMemory(_)) => {
                evaluation
                    .global_memory_bindings
                    .iter()
                    .any(|binding| binding.instruction_pcs.contains(&instruction.pc))
            }
            Some(InstructionControl::BufferMemory(_)) => {
                evaluation
                    .global_memory_bindings
                    .iter()
                    .any(|binding| binding.instruction_pcs.contains(&instruction.pc))
                    || (allow_vertex_inputs
                        && evaluation
                            .vertex_inputs
                            .iter()
                            .any(|binding| binding.pc == instruction.pc))
            }
            None if matches!(
                instruction.encoding,
                ShaderEncoding::Smrd
                    | ShaderEncoding::Smem
                    | ShaderEncoding::Mubuf
                    | ShaderEncoding::Mtbuf
                    | ShaderEncoding::Flat
                    | ShaderEncoding::Mimg
            ) =>
            {
                false
            }
            _ => true,
        }
    })
}

fn append_evaluation_ranges(
    ranges: &mut Vec<DrawFootprintRange>,
    snapshot: &ShaderSnapshot,
    evaluation: &ShaderEvaluation,
    allow_vertex_inputs: bool,
) -> Result<(), SharedEventDrawReject> {
    if !program_memory_is_completely_covered(snapshot, evaluation, allow_vertex_inputs) {
        return Err(SharedEventDrawReject::ShaderCoverage);
    }
    for binding in &evaluation.global_memory_bindings {
        let binding = effective_global_memory_binding(evaluation, binding);
        if binding.base_address == 0 {
            if binding.byte_count <= 4 {
                continue;
            }
            return Err(SharedEventDrawReject::Unresolved);
        }
        let byte_count =
            u64::try_from(binding.byte_count).map_err(|_| SharedEventDrawReject::Unresolved)?;
        let proven_read_only = !binding.writable
            && !binding.instruction_pcs.is_empty()
            && binding.instruction_pcs.iter().all(|pc| {
                snapshot
                    .program
                    .instructions
                    .iter()
                    .find(|instruction| instruction.pc == *pc)
                    .and_then(|instruction| instruction.opcode.known_name())
                    .is_some_and(|opcode| !opcode.contains("Store") && !opcode.contains("Atomic"))
            });
        append_range(
            ranges,
            if proven_read_only {
                DrawFootprintKind::GlobalRead
            } else {
                DrawFootprintKind::GlobalWrite
            },
            binding.base_address,
            byte_count,
        )?;
    }
    for &(address, _) in &evaluation.scalar_memory_reads {
        append_range(ranges, DrawFootprintKind::ScalarEvaluationRead, address, 4)?;
    }
    for binding in &evaluation.image_bindings {
        if image_resource_descriptor_is_null(&binding.resource_descriptor) {
            continue;
        }
        let descriptor = decode_image_descriptor_without_diagnostics(&binding.resource_descriptor)
            .map_err(|_| SharedEventDrawReject::Unresolved)?;
        let (_, byte_count) =
            sampled_texture_raw_range(descriptor).map_err(|_| SharedEventDrawReject::Unresolved)?;
        let read_only_sample = !is_storage_image_opcode(binding.opcode)
            && !is_writable_image_opcode(binding.opcode)
            && !image_is_comparison(binding.opcode);
        append_image_range(
            ranges,
            if read_only_sample {
                DrawFootprintKind::ImageRead
            } else {
                DrawFootprintKind::ImageWrite
            },
            descriptor,
            u64::try_from(byte_count).map_err(|_| SharedEventDrawReject::Unresolved)?,
        )?;
    }
    Ok(())
}

fn direct_draw_footprint(
    draw: &IndexedDraw,
    diagnostics_are_host_effectful: bool,
) -> Result<DirectDrawFootprint, SharedEventDrawReject> {
    if draw.indirect_arguments_address.is_some() {
        return Err(SharedEventDrawReject::Indirect);
    }
    if draw.indexed && draw.index_snapshot.is_none() {
        return Err(SharedEventDrawReject::Other);
    }
    if diagnostics_are_host_effectful {
        return Err(SharedEventDrawReject::Diagnostic);
    }

    let depth =
        decode_depth_target_without_diagnostics(draw).map_err(|_| SharedEventDrawReject::Other)?;
    let color_control = register_value(&draw.context_registers, CB_COLOR_CONTROL)
        .unwrap_or(DEFAULT_CB_COLOR_CONTROL);
    let color_operation =
        decode_color_operation_mode(color_control).map_err(|_| SharedEventDrawReject::Other)?;
    if !matches!(
        color_operation,
        ColorOperationMode::Normal | ColorOperationMode::Disable
    ) || color_operation_is_raster_noop(color_operation, depth.is_some())
    {
        return Err(SharedEventDrawReject::SpecialOperation);
    }

    let targets = decode_render_targets(draw).map_err(|_| {
        if draw.pixel_shader.is_none() {
            SharedEventDrawReject::ShaderCoverage
        } else {
            SharedEventDrawReject::Other
        }
    })?;
    if targets.is_empty() && depth.is_none() {
        return Err(SharedEventDrawReject::Other);
    }

    let export_snapshot = draw
        .export_shader
        .as_ref()
        .ok_or(SharedEventDrawReject::ShaderCoverage)?;
    let export_evaluation = draw
        .export_shader_evaluation
        .as_deref()
        .ok_or(SharedEventDrawReject::ShaderCoverage)?;
    let pixel = match (
        draw.pixel_shader.as_ref(),
        draw.pixel_shader_evaluation.as_deref(),
    ) {
        (Some(snapshot), Some(evaluation)) => Some((snapshot, evaluation)),
        (None, None) if targets.is_empty() && depth.is_some() => None,
        _ => return Err(SharedEventDrawReject::ShaderCoverage),
    };

    let mut ranges = Vec::new();
    for target in targets {
        let byte_count = resident_texture_raw_byte_count(
            target.texture,
            target.texture_type,
            target.layer_count,
            target.mip_level_count,
        )
        .map_err(|_| SharedEventDrawReject::Unresolved)?;
        append_range(
            &mut ranges,
            DrawFootprintKind::ColorAttachment,
            target.texture.address,
            byte_count,
        )?;
    }
    if let Some(depth) = depth {
        let texture_type = if depth.layer_count == 1 { 9 } else { 13 };
        let byte_count =
            resident_texture_raw_byte_count(depth.texture, texture_type, depth.layer_count, 1)
                .map_err(|_| SharedEventDrawReject::Unresolved)?;
        append_range(
            &mut ranges,
            DrawFootprintKind::DepthAttachment,
            depth.texture.address,
            byte_count,
        )?;
    }
    append_evaluation_ranges(&mut ranges, export_snapshot, export_evaluation, true)?;
    if let Some((snapshot, evaluation)) = pixel {
        append_evaluation_ranges(&mut ranges, snapshot, evaluation, false)?;
    }
    Ok(DirectDrawFootprint::from_ranges(ranges))
}

fn overlap_rejection(kind: SharedEventWriteKind) -> SharedEventDrawReject {
    match kind {
        SharedEventWriteKind::ResidentDestination => SharedEventDrawReject::OverlapResident,
        SharedEventWriteKind::DirectScanoutTarget => SharedEventDrawReject::OverlapScanout,
        SharedEventWriteKind::CopiedGlobal => SharedEventDrawReject::OverlapCopiedGlobal,
        SharedEventWriteKind::ZeroCopyGlobal => SharedEventDrawReject::OverlapZeroCopyGlobal,
        SharedEventWriteKind::TerminalGuestWrite => SharedEventDrawReject::OverlapTerminalWrite,
        SharedEventWriteKind::InternalResidentDestination => {
            SharedEventDrawReject::OverlapInternalResident
        }
    }
}

fn retained_resident_version_is_current(
    residents: &super::super::ResidentTextureMap,
    version: &super::super::SharedEventResidentTextureVersion,
) -> bool {
    let Some(resident) = residents.get(&version.descriptor) else {
        return false;
    };
    // A guest render-target version requires the resident to still be GPU-authored; an internal
    // relayout version (plan A2) does not — the relayout carried clean guest content, and the
    // watch-token check below is what proves the CPU has not touched it since.
    let requires_gpu_authored = version.kind == SharedEventWriteKind::ResidentDestination;
    if resident.texture.as_ptr() != version.texture.as_ptr()
        || resident.content_version != version.content_version
        || resident.texture_type != version.texture_type
        || resident.layer_count != version.layer_count
        || resident.mip_level_count != version.mip_level_count
        || resident.span != Some(version.span)
        || (requires_gpu_authored && !resident.gpu_written())
        || !resident.serves_reads()
    {
        return false;
    }

    // Phase 1G: coarse write-watch dirty is enough. Self-readback generations are deleted with
    // CPU writeback — a clean watch means the guest has not stored since Metal authored.
    if resident.write_watch.retirement().is_some()
        || resident.write_watch.is_dirty()
        || !resident.write_watch.dirty_page_ranges().is_empty()
        || resident.write_watch.content_validation_pending()
    {
        return false;
    }
    true
}

fn direct_draw_dependencies_are_gpu_orderable(
    consumer_queue_id: u64,
    dependencies: &[SharedEventFence],
    checkpoints: &HashMap<SharedEventProducerId, SharedEventSignalCheckpoint>,
    footprint: &DirectDrawFootprint,
    immutable_global_available: impl Fn(u64, u64) -> bool,
    resident_version_is_current: impl Fn(&super::super::SharedEventResidentTextureVersion) -> bool,
) -> Result<(DirectDrawAdmission, Vec<SharedEventDrawTextureOverride>), SharedEventDrawReject> {
    let mut admission = DirectDrawAdmission::Disjoint;
    let mut texture_overrides = Vec::<SharedEventDrawTextureOverride>::new();
    let mut dependencies = dependencies.to_vec();
    dependencies.sort_unstable_by_key(|fence| (fence.producer().queue_id(), fence.value()));
    for fence in dependencies {
        let checkpoint = checkpoints
            .get(&fence.producer())
            .filter(|checkpoint| checkpoint.fence == fence)
            .ok_or(SharedEventDrawReject::IncompleteProducer)?;
        if !checkpoint.footprint_complete {
            return Err(SharedEventDrawReject::IncompleteProducer);
        }
        for &producer in &checkpoint.pending_write_footprint {
            for &consumer in footprint
                .ranges
                .iter()
                .filter(|&&consumer| consumer.overlaps_producer(producer))
            {
                if producer.kind == SharedEventWriteKind::CopiedGlobal
                    && consumer.kind == DrawFootprintKind::GlobalRead
                    && immutable_global_available(consumer.guest_address, consumer.byte_count)
                {
                    admission = DirectDrawAdmission::VersionedGlobal;
                    continue;
                }
                if matches!(
                    producer.kind,
                    SharedEventWriteKind::ResidentDestination
                        | SharedEventWriteKind::InternalResidentDestination
                ) && consumer.kind == DrawFootprintKind::ImageRead
                    && fence.producer().queue_id() == consumer_queue_id
                    && let Some(sampled) = consumer.image
                    && let Some(version) = checkpoint
                        .resident_texture_versions
                        .iter()
                        .filter(|version| {
                            version.kind == producer.kind
                                && version.span.backing() == Some(producer.mapping)
                                && version.span.len() == producer.byte_count
                                && version.descriptor == sampled.guest
                                && version.texture_type == sampled.texture_type
                                && version.layer_count == sampled.layer_count
                                && version.mip_level_count == sampled.mip_level_count.max(1)
                                && sampled.base_mip_level == 0
                                && consumer.mapping == producer.mapping
                                && consumer.byte_count == producer.byte_count
                        })
                        .max_by_key(|version| version.content_version)
                    && resident_version_is_current(version)
                {
                    let retained = SharedEventDrawTextureOverride {
                        sampled,
                        texture: version.texture.clone(),
                    };
                    if let Some(existing) = texture_overrides.iter_mut().find(|existing| {
                        existing.sampled.guest == sampled.guest
                            && existing.sampled.destination_select == sampled.destination_select
                            && existing.sampled.base_mip_level == sampled.base_mip_level
                            && existing.sampled.mip_level_count == sampled.mip_level_count
                            && existing.sampled.texture_type == sampled.texture_type
                            && existing.sampled.layer_count == sampled.layer_count
                    }) {
                        *existing = retained;
                    } else {
                        texture_overrides.push(retained);
                    }
                    admission = DirectDrawAdmission::ResidentImage;
                    continue;
                }
                return Err(overlap_rejection(producer.kind));
            }
        }
    }
    Ok((admission, texture_overrides))
}

fn first_physical_overlap(
    state: &GuestQueueBatchState,
    footprint: &DirectDrawFootprint,
) -> Option<(PhysicalFootprintRange, DrawFootprintRange)> {
    state
        .resource_dependencies
        .values()
        .flat_map(|checkpoint| &checkpoint.pending_write_footprint)
        .find_map(|&producer| {
            footprint
                .ranges
                .iter()
                .copied()
                .find(|&consumer| consumer.overlaps_producer(producer))
                .map(|consumer| (producer, consumer))
        })
}

fn describe_consumer_bytes(footprint: &DirectDrawFootprint) -> String {
    let mut bytes = [0_u64; DrawFootprintKind::COUNT];
    for range in footprint.ranges.iter() {
        bytes[range.kind.index()] = bytes[range.kind.index()].saturating_add(range.byte_count);
    }
    [
        DrawFootprintKind::ColorAttachment,
        DrawFootprintKind::DepthAttachment,
        DrawFootprintKind::GlobalRead,
        DrawFootprintKind::GlobalWrite,
        DrawFootprintKind::ScalarEvaluationRead,
        DrawFootprintKind::ImageRead,
        DrawFootprintKind::ImageWrite,
    ]
    .into_iter()
    .map(|kind| format!("{}:{}", kind.label(), bytes[kind.index()]))
    .collect::<Vec<_>>()
    .join(",")
}

fn producer_kind_index(kind: SharedEventWriteKind) -> usize {
    match kind {
        SharedEventWriteKind::ResidentDestination => 0,
        SharedEventWriteKind::DirectScanoutTarget => 1,
        SharedEventWriteKind::CopiedGlobal => 2,
        SharedEventWriteKind::ZeroCopyGlobal => 3,
        SharedEventWriteKind::TerminalGuestWrite => 4,
        SharedEventWriteKind::InternalResidentDestination => 5,
    }
}

fn describe_producer_bytes(state: Option<&GuestQueueBatchState>) -> String {
    let mut bytes = [0_u64; 6];
    for range in state
        .into_iter()
        .flat_map(|state| state.resource_dependencies.values())
        .flat_map(|checkpoint| &checkpoint.pending_write_footprint)
    {
        let index = producer_kind_index(range.kind);
        bytes[index] = bytes[index].saturating_add(range.byte_count);
    }
    [
        "resident",
        "scanout",
        "copied-global",
        "zero-copy-global",
        "terminal-write",
        "internal-resident",
    ]
    .into_iter()
    .zip(bytes)
    .map(|(kind, bytes)| format!("{kind}:{bytes}"))
    .collect::<Vec<_>>()
    .join(",")
}

impl MetalPresenter {
    pub(super) fn observe_shared_event_direct_draw_preflight(
        &mut self,
        consumer_queue_id: u64,
        dependencies: &[SharedEventFence],
        draw: &IndexedDraw,
    ) -> Option<SharedEventDirectDrawPlan> {
        self.statistics.shared_event_draw_attempts =
            self.statistics.shared_event_draw_attempts.saturating_add(1);
        let footprint = direct_draw_footprint(draw, draw_diagnostics_are_host_effectful());
        let decision = match &footprint {
            Err(reject) => Err(*reject),
            Ok(footprint) => self
                .guest_queue_batches
                .get(&consumer_queue_id)
                .ok_or(SharedEventDrawReject::IncompleteProducer)
                .and_then(|state| {
                    direct_draw_dependencies_are_gpu_orderable(
                        consumer_queue_id,
                        dependencies,
                        &state.resource_dependencies,
                        footprint,
                        |address, byte_count| {
                            self.shared_event_dependencies_supply_draw_global_range(
                                consumer_queue_id,
                                address,
                                byte_count,
                            )
                        },
                        |version| {
                            retained_resident_version_is_current(
                                &self.resident_guest_textures,
                                version,
                            )
                        },
                    )
                }),
        };
        match decision {
            Ok((DirectDrawAdmission::Disjoint, _)) => {
                self.statistics.shared_event_draw_disjoint_candidates = self
                    .statistics
                    .shared_event_draw_disjoint_candidates
                    .saturating_add(1);
            }
            Ok((DirectDrawAdmission::VersionedGlobal, _)) => {
                self.statistics
                    .shared_event_draw_versioned_global_candidates = self
                    .statistics
                    .shared_event_draw_versioned_global_candidates
                    .saturating_add(1);
            }
            Ok((DirectDrawAdmission::ResidentImage, _)) => {
                self.statistics.shared_event_draw_resident_image_candidates = self
                    .statistics
                    .shared_event_draw_resident_image_candidates
                    .saturating_add(1);
            }
            Err(reject) => {
                self.statistics.shared_event_draw_rejections[reject.index()] =
                    self.statistics.shared_event_draw_rejections[reject.index()].saturating_add(1);
            }
        }

        // The overlap-only spelling deliberately avoids the broad REMU_TRACE_ diagnostic gate:
        // it classifies the ordinary preflight without making the draw footprint unavailable.
        // Use it with admission disabled; the bounded log itself is not a performance run.
        let trace_draws = std::env::var_os("REMU_TRACE_SHARED_EVENT_DRAWS")
            .is_some_and(|value| value == "1")
            || std::env::var_os("REMU_SHARED_EVENT_DRAW_OVERLAPS")
                .is_some_and(|value| value == "1");
        if trace_draws {
            static TRACES: AtomicUsize = AtomicUsize::new(0);
            let sequence = TRACES.fetch_add(1, Ordering::Relaxed) + 1;
            if sequence <= 16 || sequence.is_power_of_two() {
                let dependency_text = dependencies
                    .iter()
                    .map(|fence| format!("q{}:{}", fence.producer().queue_id(), fence.value()))
                    .collect::<Vec<_>>()
                    .join(",");
                let state = self.guest_queue_batches.get(&consumer_queue_id);
                let producer_bytes = state.map_or(0, |state| {
                    state
                        .resource_dependencies
                        .values()
                        .flat_map(|checkpoint| &checkpoint.pending_write_footprint)
                        .fold(0_u64, |bytes, range| bytes.saturating_add(range.byte_count))
                });
                let overlap = footprint.as_ref().ok().and_then(|footprint| {
                    state.and_then(|state| first_physical_overlap(state, footprint))
                });
                let consumer_kinds = footprint
                    .as_ref()
                    .map_or_else(|_| "unavailable".to_owned(), describe_consumer_bytes);
                let producer_kinds = describe_producer_bytes(state);
                eprintln!(
                    "[REMu][METAL][SHARED-DRAW] sequence={sequence} draw={} consumer=q{consumer_queue_id} dependencies=[{dependency_text}] result={} consumer-bytes={} consumer-kinds=[{consumer_kinds}] producer-bytes={producer_bytes} producer-kinds=[{producer_kinds}] overlap={overlap:?}",
                    draw.sequence,
                    decision
                        .as_ref()
                        .map_or_else(|reject| reject.label(), |(admission, _)| admission.label()),
                    footprint
                        .as_ref()
                        .map_or(0, DirectDrawFootprint::byte_count),
                );
            }
        }
        decision
            .ok()
            .map(|(_, texture_overrides)| SharedEventDirectDrawPlan { texture_overrides })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use metal::foreign_types::ForeignType;
    use metal::{Device, MTLStorageMode, MTLTextureType, MTLTextureUsage, TextureDescriptor};
    use remu_gpu::{GpuRegister, ShaderEvaluationSnapshot, ShaderSnapshot};
    use remu_memory::{
        GUEST_PAGE_SIZE, GuestAddressSpace, GuestMapPlacement, GuestPhysicalBacking,
        GuestProtection,
    };
    use remu_shader::{
        BufferMemoryControl, ExportControl, GlobalMemoryBinding, GlobalMemoryControl, ImageBinding,
        ImageControl, Instruction, Opcode, Program, ScalarMemoryControl, ShaderEvaluation,
        VertexInputBinding,
    };

    fn mapped_space(pages: u64) -> (GuestAddressSpace, u64) {
        let mut space =
            GuestAddressSpace::new(pages * GUEST_PAGE_SIZE).expect("guest address space");
        let backing =
            GuestPhysicalBacking::new(pages * GUEST_PAGE_SIZE).expect("guest physical backing");
        let base = space
            .map_shared(
                GuestMapPlacement::Fixed(space.base()),
                pages * GUEST_PAGE_SIZE,
                GUEST_PAGE_SIZE,
                GuestProtection::READ_WRITE,
                &backing,
                0,
            )
            .expect("complete shared mapping")
            .get();
        (space, base)
    }

    fn empty_evaluation() -> ShaderEvaluation {
        ShaderEvaluation {
            initial_scalar_registers: Vec::new().into(),
            scalar_registers: Vec::new().into(),
            scalar_memory_reads: Vec::new(),
            image_bindings: Vec::new(),
            global_memory_bindings: Vec::new(),
            compute_system_registers: None,
            vertex_inputs: Vec::new(),
        }
    }

    fn mutate_evaluation(
        snapshot: &mut ShaderEvaluationSnapshot,
        mutate: impl FnOnce(&mut ShaderEvaluation),
    ) {
        let mut evaluation = Arc::clone(snapshot.evaluation());
        mutate(Arc::make_mut(&mut evaluation));
        *snapshot = evaluation.into();
    }

    fn snapshot(address: u64, instructions: Vec<Instruction>) -> ShaderSnapshot {
        ShaderSnapshot {
            header_address: address,
            program: Arc::new(Program::new(address, instructions)),
            supplemental_programs: Arc::from([]),
            metadata: None,
        }
    }

    fn instruction(
        pc: u32,
        encoding: ShaderEncoding,
        opcode: &'static str,
        control: Option<InstructionControl>,
    ) -> Instruction {
        Instruction {
            pc,
            encoding,
            opcode: Opcode::Known(opcode),
            words: Vec::new(),
            sources: Vec::new(),
            destinations: Vec::new(),
            control,
        }
    }

    fn export_control(target: u32) -> InstructionControl {
        InstructionControl::Export(ExportControl {
            target,
            enable_mask: 0xf,
            compressed: false,
            done: true,
            valid_mask: false,
        })
    }

    fn image_control() -> ImageControl {
        ImageControl {
            dmask: 0xf,
            vector_address: 0,
            address_registers: vec![0, 1],
            vector_data: 4,
            scalar_resource: 0,
            scalar_sampler: 8,
            dimension: 1,
            is_array: false,
            glc: false,
            slc: false,
            a16: false,
            d16: false,
            unorm: false,
        }
    }

    fn image_descriptor(address: u64, width: u32, height: u32) -> Vec<u32> {
        let encoded = address >> 8;
        let width = width - 1;
        let height = height - 1;
        vec![
            u32::try_from(encoded).expect("test image address low bits fit u32"),
            ((encoded >> 32) as u32 & 0xff) | 0x0380_0000 | ((width & 3) << 30),
            (width >> 2) | (height << 14),
            0x9000_0f2e,
            width,
            0,
            0,
            0,
        ]
    }

    fn append_color_target(
        registers: &mut Arc<[GpuRegister]>,
        slot: u32,
        address: u64,
        width: u32,
        height: u32,
    ) {
        let primary = slot * super::super::super::CB_COLOR_REGISTER_STRIDE;
        extend_registers(
            registers,
            [
                GpuRegister {
                    offset: super::super::super::decode::CB_COLOR0_BASE + primary,
                    value: u32::try_from(address >> 8)
                        .expect("test color-target address low bits fit u32"),
                },
                GpuRegister {
                    offset: super::super::super::decode::CB_COLOR0_VIEW + primary,
                    value: 0,
                },
                GpuRegister {
                    offset: super::super::super::decode::CB_COLOR0_INFO + primary,
                    value: 10 << 2,
                },
                GpuRegister {
                    offset: super::super::super::decode::CB_COLOR0_ATTRIB + primary,
                    value: 0,
                },
                GpuRegister {
                    offset: super::super::super::decode::CB_COLOR0_BASE_EXT + slot,
                    value: ((address >> 40) & 0xff) as u32,
                },
                GpuRegister {
                    offset: super::super::super::decode::CB_COLOR0_ATTRIB2 + slot,
                    value: ((width - 1) << 14) | (height - 1),
                },
                GpuRegister {
                    offset: super::super::super::decode::CB_COLOR0_ATTRIB3 + slot,
                    value: u32::from(height > 1) << 24,
                },
            ],
        );
    }

    fn mutate_registers(
        registers: &mut Arc<[GpuRegister]>,
        mutate: impl FnOnce(&mut Vec<GpuRegister>),
    ) {
        let mut owned = registers.to_vec();
        mutate(&mut owned);
        *registers = owned.into();
    }

    fn push_register(registers: &mut Arc<[GpuRegister]>, register: GpuRegister) {
        mutate_registers(registers, |owned| owned.push(register));
    }

    fn extend_registers<const N: usize>(
        registers: &mut Arc<[GpuRegister]>,
        extension: [GpuRegister; N],
    ) {
        mutate_registers(registers, |owned| owned.extend(extension));
    }

    #[allow(clippy::too_many_lines)]
    fn complete_draw(base: u64) -> IndexedDraw {
        let mut draw = IndexedDraw {
            sequence: 7,
            indexed: true,
            indirect_arguments_address: None,
            instance_count: 1,
            index_buffer_address: 0,
            index_resource_address: 0,
            index_buffer_count: 3,
            index_size: 4,
            index_offset: 0,
            base_vertex: 0,
            index_count: 3,
            index_snapshot: Some(remu_gpu::IndexSnapshot::from([0, 1, 2])),
            draw_flags: 0,
            primitive_type: 4,
            export_shader_address: 0x1000,
            pixel_shader_address: 0x2000,
            export_shader_prefix: [0; 4],
            pixel_shader_prefix: [0; 4],
            export_shader: None,
            pixel_shader: None,
            export_shader_evaluation: None,
            pixel_shader_evaluation: None,
            color_target_address: base,
            scanout_target: None,
            context_registers: Arc::from([]),
            shader_registers: Arc::from([]),
            user_config_registers: Arc::from([]),
        };
        append_color_target(&mut draw.context_registers, 0, base, 64, 64);
        append_color_target(
            &mut draw.context_registers,
            1,
            base + GUEST_PAGE_SIZE,
            64,
            64,
        );
        push_register(
            &mut draw.context_registers,
            GpuRegister {
                offset: super::super::super::CB_TARGET_MASK,
                value: 0xff,
            },
        );

        let depth = base + 2 * GUEST_PAGE_SIZE;
        let depth_base =
            u32::try_from(depth >> 8).expect("test depth-target address low bits fit u32");
        let depth_base_hi = ((depth >> 40) & 0xff) as u32;
        extend_registers(
            &mut draw.context_registers,
            [
                GpuRegister {
                    offset: super::super::super::DB_DEPTH_SIZE_XY,
                    value: 63 | (63 << 16),
                },
                GpuRegister {
                    offset: super::super::super::DB_Z_INFO,
                    value: 3,
                },
                GpuRegister {
                    offset: super::super::super::decode::DB_Z_READ_BASE,
                    value: depth_base,
                },
                GpuRegister {
                    offset: super::super::super::decode::DB_Z_WRITE_BASE,
                    value: depth_base,
                },
                GpuRegister {
                    offset: super::super::super::decode::DB_Z_READ_BASE_HI,
                    value: depth_base_hi,
                },
                GpuRegister {
                    offset: super::super::super::decode::DB_Z_WRITE_BASE_HI,
                    value: depth_base_hi,
                },
                GpuRegister {
                    offset: super::super::super::DB_DEPTH_CONTROL,
                    value: 0x66,
                },
            ],
        );
        Arc::make_mut(&mut draw.context_registers).sort_unstable_by_key(|register| register.offset);

        let global = base + 3 * GUEST_PAGE_SIZE;
        let writable_global = base + 4 * GUEST_PAGE_SIZE;
        let export_image = base + 5 * GUEST_PAGE_SIZE;
        let pixel_image = base + 6 * GUEST_PAGE_SIZE;
        let global_control = GlobalMemoryControl {
            dword_count: 1,
            vector_address: 0,
            vector_data: 0,
            vector_destination: 0,
            scalar_address: Some(0),
            offset_bytes: 0,
            dlc: false,
            lds: false,
            glc: false,
            slc: false,
        };
        let buffer_control = BufferMemoryControl {
            dword_count: 1,
            vector_address: 0,
            vector_data: 0,
            scalar_resource: 0,
            instruction_format: None,
            offset_bytes: 0,
            index_enabled: false,
            offset_enabled: false,
            dlc: false,
            lds: false,
            glc: false,
            slc: false,
            tfe: false,
        };
        let scalar_control = ScalarMemoryControl {
            destination_count: 1,
            immediate_offset_bytes: 0,
            dynamic_offset_register: None,
        };
        draw.export_shader = Some(snapshot(
            draw.export_shader_address,
            vec![
                instruction(
                    4,
                    ShaderEncoding::Flat,
                    "GlobalLoadDword",
                    Some(InstructionControl::GlobalMemory(global_control)),
                ),
                instruction(
                    8,
                    ShaderEncoding::Mubuf,
                    "BufferLoadDword",
                    Some(InstructionControl::BufferMemory(buffer_control)),
                ),
                instruction(
                    12,
                    ShaderEncoding::Smem,
                    "SLoadDword",
                    Some(InstructionControl::ScalarMemory(scalar_control)),
                ),
                instruction(
                    16,
                    ShaderEncoding::Mimg,
                    "ImageSample",
                    Some(InstructionControl::Image(image_control())),
                ),
            ],
        ));
        draw.pixel_shader = Some(snapshot(
            draw.pixel_shader_address,
            vec![
                instruction(
                    20,
                    ShaderEncoding::Mimg,
                    "ImageStore",
                    Some(InstructionControl::Image(image_control())),
                ),
                instruction(
                    24,
                    ShaderEncoding::Flat,
                    "GlobalStoreDword",
                    Some(InstructionControl::GlobalMemory(global_control)),
                ),
                instruction(28, ShaderEncoding::Exp, "Exp", Some(export_control(0))),
                instruction(32, ShaderEncoding::Exp, "Exp", Some(export_control(1))),
            ],
        ));
        let mut export_evaluation = empty_evaluation();
        export_evaluation.scalar_memory_reads.push((global + 64, 1));
        export_evaluation
            .global_memory_bindings
            .push(GlobalMemoryBinding {
                scalar_address: 0,
                base_address: global,
                instruction_pcs: vec![4, 12].into(),
                byte_count: 64,
                data: vec![0; 64],
                writable: false,
                write_back_to_guest: false,
            });
        export_evaluation.vertex_inputs.push(VertexInputBinding {
            pc: 8,
            location: 0,
            component_count: 4,
            data_format: 10,
            number_format: 0,
            base_address: 0,
            stride: 16,
            offset_bytes: 0,
            data: vec![0; 48],
        });
        export_evaluation.image_bindings.push(ImageBinding { descriptor_index_is_runtime: false,
            pc: 16,
            opcode: Opcode::Known("ImageSample"),
            control: image_control(),
            resource_descriptor: image_descriptor(export_image, 32, 32).into(),
            sampler_descriptor: vec![0; 4].into(),
            mip_level: None,
        });
        let mut pixel_evaluation = empty_evaluation();
        pixel_evaluation
            .scalar_memory_reads
            .push((writable_global + 64, 2));
        pixel_evaluation
            .global_memory_bindings
            .push(GlobalMemoryBinding {
                scalar_address: 4,
                base_address: writable_global,
                instruction_pcs: vec![24].into(),
                byte_count: 64,
                data: vec![0; 64],
                writable: true,
                write_back_to_guest: true,
            });
        pixel_evaluation.image_bindings.push(ImageBinding { descriptor_index_is_runtime: false,
            pc: 20,
            opcode: Opcode::Known("ImageStore"),
            control: image_control(),
            resource_descriptor: image_descriptor(pixel_image, 32, 32).into(),
            sampler_descriptor: Vec::new().into(),
            mip_level: None,
        });
        draw.export_shader_evaluation = Some(Arc::new(export_evaluation).into());
        draw.pixel_shader_evaluation = Some(Arc::new(pixel_evaluation).into());
        draw
    }

    #[test]
    fn complete_direct_draw_captures_all_attachment_and_shader_resource_classes() {
        let (_space, base) = mapped_space(16);
        let footprint = direct_draw_footprint(&complete_draw(base), false)
            .expect("complete mapped draw has a footprint");
        for kind in [
            DrawFootprintKind::ColorAttachment,
            DrawFootprintKind::DepthAttachment,
            DrawFootprintKind::GlobalRead,
            DrawFootprintKind::GlobalWrite,
            DrawFootprintKind::ScalarEvaluationRead,
            DrawFootprintKind::ImageRead,
            DrawFootprintKind::ImageWrite,
        ] {
            assert!(
                footprint.ranges.iter().any(|range| range.kind == kind),
                "footprint omitted {kind:?}"
            );
        }
        assert_eq!(
            footprint
                .ranges
                .iter()
                .filter(|range| range.kind == DrawFootprintKind::ColorAttachment)
                .fold(0_u64, |bytes, range| bytes + range.byte_count),
            2 * GUEST_PAGE_SIZE,
            "MRT0 and MRT1 remain covered after adjacent physical intervals coalesce"
        );
    }

    #[test]
    fn direct_draw_global_read_proof_rejects_store_opcodes_and_keeps_bindings_exact() {
        let (_space, base) = mapped_space(16);
        let mut draw = complete_draw(base);
        let mut pixel = draw
            .pixel_shader_evaluation
            .as_deref()
            .expect("pixel evaluation")
            .clone();
        pixel.global_memory_bindings[0].writable = false;
        draw.pixel_shader_evaluation = Some(Arc::new(pixel).into());
        let footprint = direct_draw_footprint(&draw, false).expect("complete draw footprint");
        assert!(
            footprint
                .ranges
                .iter()
                .any(|range| range.kind == DrawFootprintKind::GlobalWrite),
            "a false writable flag cannot make a decoded GlobalStore forwardable"
        );

        let first = DrawFootprintRange::new(DrawFootprintKind::GlobalRead, base, 64).unwrap();
        let second = DrawFootprintRange::new(DrawFootprintKind::GlobalRead, base + 64, 64).unwrap();
        let exact = DirectDrawFootprint::from_ranges(vec![first, second]);
        assert_eq!(
            exact.ranges.len(),
            2,
            "adjacent immutable bindings need independent containment proofs"
        );
    }

    #[test]
    fn fragmentless_depth_only_draw_is_structurally_complete() {
        let (_space, base) = mapped_space(16);
        let mut draw = complete_draw(base);
        mutate_registers(&mut draw.context_registers, |registers| {
            registers.retain(|register| {
                !matches!(
                    register.offset,
                    super::super::super::CB_TARGET_MASK
                        | super::super::super::decode::CB_COLOR0_BASE
                        | super::super::super::decode::CB_COLOR0_VIEW
                        | super::super::super::decode::CB_COLOR0_INFO
                        | super::super::super::decode::CB_COLOR0_ATTRIB
                        | super::super::super::decode::CB_COLOR0_ATTRIB2
                        | super::super::super::decode::CB_COLOR0_ATTRIB3
                ) && register.offset
                    < super::super::super::decode::CB_COLOR0_BASE
                        + super::super::super::CB_COLOR_REGISTER_STRIDE
            })
        });
        draw.pixel_shader = None;
        draw.pixel_shader_evaluation = None;
        let footprint = direct_draw_footprint(&draw, false)
            .expect("fragmentless depth-only direct draw is complete");
        assert!(
            footprint
                .ranges
                .iter()
                .any(|range| range.kind == DrawFootprintKind::DepthAttachment)
        );
    }

    #[test]
    fn opt_in_admission_keeps_structural_shader_diagnostic_and_special_rejections() {
        let (_space, base) = mapped_space(16);
        let draw = complete_draw(base);
        let mut cases = Vec::new();

        let mut indirect = draw.clone();
        indirect.indirect_arguments_address = Some(base + 7 * GUEST_PAGE_SIZE);
        cases.push((indirect, SharedEventDrawReject::Indirect));

        let mut no_indices = draw.clone();
        no_indices.index_snapshot = None;
        cases.push((no_indices, SharedEventDrawReject::Other));

        let mut supplemental = draw.clone();
        supplemental
            .export_shader
            .as_mut()
            .expect("export snapshot")
            .supplemental_programs = Arc::from([Program::new(0x3000, Vec::new())]);
        cases.push((supplemental, SharedEventDrawReject::ShaderCoverage));

        let mut missing_image = draw.clone();
        mutate_evaluation(
            missing_image
                .export_shader_evaluation
                .as_mut()
                .expect("export evaluation"),
            |evaluation| evaluation.image_bindings.clear(),
        );
        cases.push((missing_image, SharedEventDrawReject::ShaderCoverage));

        let mut missing_export = draw.clone();
        missing_export.export_shader_evaluation = None;
        cases.push((missing_export, SharedEventDrawReject::ShaderCoverage));

        let mut missing_pixel = draw.clone();
        missing_pixel.pixel_shader_evaluation = None;
        cases.push((missing_pixel, SharedEventDrawReject::ShaderCoverage));

        for operation in [2, 3, 5] {
            let mut special = draw.clone();
            push_register(
                &mut special.context_registers,
                GpuRegister {
                    offset: super::super::super::CB_COLOR_CONTROL,
                    value: operation << 4,
                },
            );
            Arc::make_mut(&mut special.context_registers)
                .sort_unstable_by_key(|register| register.offset);
            cases.push((special, SharedEventDrawReject::SpecialOperation));
        }

        cases.push((draw.clone(), SharedEventDrawReject::Diagnostic));

        let last = cases.len() - 1;
        for (index, (draw, expected)) in cases.into_iter().enumerate() {
            let diagnostic = index == last;
            assert_eq!(direct_draw_footprint(&draw, diagnostic), Err(expected));
        }
    }

    #[test]
    fn null_stubs_are_ignored_but_nonnull_unmapped_resources_reject() {
        let (_space, base) = mapped_space(16);
        let mut draw = complete_draw(base);
        mutate_evaluation(
            draw.export_shader_evaluation
                .as_mut()
                .expect("export evaluation"),
            |evaluation| {
                evaluation.global_memory_bindings.push(GlobalMemoryBinding {
                    scalar_address: 99,
                    base_address: 0,
                    instruction_pcs: Vec::new().into(),
                    byte_count: 4,
                    data: vec![0; 4],
                    writable: false,
                    write_back_to_guest: false,
                });
                evaluation.image_bindings.push(ImageBinding { descriptor_index_is_runtime: false,
                    pc: 999,
                    opcode: Opcode::Known("ImageSample"),
                    control: image_control(),
                    resource_descriptor: vec![0; 8].into(),
                    sampler_descriptor: vec![0; 4].into(),
                    mip_level: None,
                });
            },
        );
        direct_draw_footprint(&draw, false).expect("true null stubs add no guest interval");

        mutate_evaluation(
            draw.export_shader_evaluation
                .as_mut()
                .expect("export evaluation"),
            |evaluation| {
                evaluation.global_memory_bindings.push(GlobalMemoryBinding {
                    scalar_address: 100,
                    base_address: 0x7fff_0000_0000,
                    instruction_pcs: Vec::new().into(),
                    byte_count: 16,
                    data: vec![0; 16],
                    writable: false,
                    write_back_to_guest: false,
                });
            },
        );
        assert_eq!(
            direct_draw_footprint(&draw, false),
            Err(SharedEventDrawReject::Unresolved)
        );

        let mut malformed_image = complete_draw(base);
        mutate_evaluation(
            malformed_image
                .export_shader_evaluation
                .as_mut()
                .expect("export evaluation"),
            |evaluation| evaluation.image_bindings[0].resource_descriptor = vec![1].into(),
        );
        assert_eq!(
            direct_draw_footprint(&malformed_image, false),
            Err(SharedEventDrawReject::Unresolved)
        );

        let mut overflowing_scalar = complete_draw(base);
        mutate_evaluation(
            overflowing_scalar
                .export_shader_evaluation
                .as_mut()
                .expect("export evaluation"),
            |evaluation| evaluation.scalar_memory_reads.push((u64::MAX - 1, 0)),
        );
        assert_eq!(
            direct_draw_footprint(&overflowing_scalar, false),
            Err(SharedEventDrawReject::Unresolved)
        );
    }

    #[test]
    fn raw_and_each_uncovered_memory_control_reject() {
        let (_space, base) = mapped_space(16);
        for pc in [4, 8, 12, 16, 20, 24] {
            let mut draw = complete_draw(base);
            if pc <= 16 {
                mutate_evaluation(
                    draw.export_shader_evaluation
                        .as_mut()
                        .expect("export evaluation"),
                    |evaluation| {
                        evaluation.global_memory_bindings.iter_mut().for_each(|binding| {
                            binding.instruction_pcs.retain(|covered| *covered != pc);
                        });
                        evaluation.vertex_inputs.retain(|binding| binding.pc != pc);
                        evaluation.image_bindings.retain(|binding| binding.pc != pc);
                    },
                );
            } else {
                mutate_evaluation(
                    draw.pixel_shader_evaluation
                        .as_mut()
                        .expect("pixel evaluation"),
                    |evaluation| {
                        evaluation.global_memory_bindings.iter_mut().for_each(|binding| {
                            binding.instruction_pcs.retain(|covered| *covered != pc);
                        });
                        evaluation.image_bindings.retain(|binding| binding.pc != pc);
                    },
                );
            }
            assert_eq!(
                direct_draw_footprint(&draw, false),
                Err(SharedEventDrawReject::ShaderCoverage),
                "uncovered instruction PC {pc:#x}"
            );
        }

        let mut raw = complete_draw(base);
        let snapshot = raw.export_shader.as_mut().expect("export snapshot");
        let program = Arc::make_mut(&mut snapshot.program);
        program.instructions[0].opcode = Opcode::Raw {
            family: remu_shader::RawOpcodeFamily::Mubuf,
            code: 0xff,
        };
        assert_eq!(
            direct_draw_footprint(&raw, false),
            Err(SharedEventDrawReject::ShaderCoverage)
        );
    }

    #[test]
    fn opt_in_admission_rejects_physical_alias_overlap_and_nonexact_checkpoints() {
        let mut space = GuestAddressSpace::new(4 * GUEST_PAGE_SIZE).expect("alias address space");
        let backing = GuestPhysicalBacking::new(GUEST_PAGE_SIZE).expect("alias backing");
        let first = space
            .map_shared(
                GuestMapPlacement::Fixed(space.base()),
                GUEST_PAGE_SIZE,
                GUEST_PAGE_SIZE,
                GuestProtection::READ_WRITE,
                &backing,
                0,
            )
            .expect("first alias")
            .get();
        let second = space
            .map_shared(
                GuestMapPlacement::Fixed(
                    space
                        .base()
                        .checked_add(2 * GUEST_PAGE_SIZE)
                        .expect("second alias address"),
                ),
                GUEST_PAGE_SIZE,
                GUEST_PAGE_SIZE,
                GuestProtection::READ_WRITE,
                &backing,
                0,
            )
            .expect("second alias")
            .get();
        let consumer = DirectDrawFootprint::from_ranges(vec![
            DrawFootprintRange::new(DrawFootprintKind::GlobalRead, second + 64, 64)
                .expect("consumer alias footprint"),
        ]);
        let producer = PhysicalFootprintRange::new(
            super::super::super::SharedEventWriteKind::ResidentDestination,
            first + 32,
            64,
            remu_memory::guest_shared_backing_identity(first + 32, 64),
        )
        .expect("producer alias footprint");
        let producer_id = SharedEventProducerId::new(41);
        let fence = SharedEventFence::new(producer_id, 7);
        let checkpoint = SharedEventSignalCheckpoint {
            fence,
            global_writeback_count: 0,
            guest_write_range_count: 0,
            memory_write_count: 0,
            global_versions: Vec::new(),
            resident_texture_versions: Vec::new(),
            unresolved_write_ranges: Vec::new(),
            pending_write_footprint: vec![producer],
            footprint_complete: true,
        };
        let mut checkpoints = HashMap::from([(producer_id, checkpoint.clone())]);
        assert_eq!(
            direct_draw_dependencies_are_gpu_orderable(
                41,
                &[fence],
                &checkpoints,
                &consumer,
                |_, _| false,
                |_| true,
            )
            .map(|(admission, _)| admission),
            Err(SharedEventDrawReject::OverlapResident)
        );

        checkpoints.get_mut(&producer_id).expect("checkpoint").fence =
            SharedEventFence::new(producer_id, 8);
        assert_eq!(
            direct_draw_dependencies_are_gpu_orderable(
                41,
                &[fence],
                &checkpoints,
                &consumer,
                |_, _| false,
                |_| true,
            )
            .map(|(admission, _)| admission),
            Err(SharedEventDrawReject::IncompleteProducer)
        );
        checkpoints.get_mut(&producer_id).expect("checkpoint").fence = fence;
        checkpoints
            .get_mut(&producer_id)
            .expect("checkpoint")
            .footprint_complete = false;
        assert_eq!(
            direct_draw_dependencies_are_gpu_orderable(
                41,
                &[fence],
                &checkpoints,
                &consumer,
                |_, _| false,
                |_| true,
            )
            .map(|(admission, _)| admission),
            Err(SharedEventDrawReject::IncompleteProducer)
        );

        let rebound_consumer = DirectDrawFootprint::from_ranges(vec![DrawFootprintRange {
            kind: DrawFootprintKind::GlobalRead,
            guest_address: first + 32,
            byte_count: 64,
            mapping: GuestSharedMappingIdentity {
                backing_id: producer.mapping.backing_id + 1,
                backing_offset: producer.mapping.backing_offset,
            },
            image: None,
        }]);
        let exact_checkpoint = SharedEventSignalCheckpoint {
            footprint_complete: true,
            ..checkpoint
        };
        let rebound_checkpoints = HashMap::from([(producer_id, exact_checkpoint)]);
        assert_eq!(
            direct_draw_dependencies_are_gpu_orderable(
                41,
                &[fence],
                &rebound_checkpoints,
                &rebound_consumer,
                |_, _| false,
                |_| true,
            )
            .map(|(admission, _)| admission),
            Ok(DirectDrawAdmission::Disjoint),
            "identical virtual addresses rebound to different captured backings are disjoint"
        );
    }

    #[test]
    fn copied_global_overlap_requires_an_exact_immutable_read_version() {
        let (_space, address) = mapped_space(2);
        let consumer = DirectDrawFootprint::from_ranges(vec![
            DrawFootprintRange::new(DrawFootprintKind::GlobalRead, address + 64, 64)
                .expect("consumer global footprint"),
        ]);
        let producer = PhysicalFootprintRange::new(
            SharedEventWriteKind::CopiedGlobal,
            address + 32,
            128,
            remu_memory::guest_shared_backing_identity(address + 32, 128),
        )
        .expect("copied global producer footprint");
        let producer_id = SharedEventProducerId::new(41);
        let fence = SharedEventFence::new(producer_id, 7);
        let checkpoint = SharedEventSignalCheckpoint {
            fence,
            global_writeback_count: 0,
            guest_write_range_count: 0,
            memory_write_count: 0,
            global_versions: Vec::new(),
            resident_texture_versions: Vec::new(),
            unresolved_write_ranges: Vec::new(),
            pending_write_footprint: vec![producer],
            footprint_complete: true,
        };
        let checkpoints = HashMap::from([(producer_id, checkpoint)]);

        assert_eq!(
            direct_draw_dependencies_are_gpu_orderable(
                41,
                &[fence],
                &checkpoints,
                &consumer,
                |candidate, byte_count| candidate == address + 64 && byte_count == 64,
                |_| true,
            )
            .map(|(admission, _)| admission),
            Ok(DirectDrawAdmission::VersionedGlobal)
        );
        assert_eq!(
            direct_draw_dependencies_are_gpu_orderable(
                41,
                &[fence],
                &checkpoints,
                &consumer,
                |_, _| false,
                |_| true,
            )
            .map(|(admission, _)| admission),
            Err(SharedEventDrawReject::OverlapCopiedGlobal)
        );

        let writable = DirectDrawFootprint::from_ranges(vec![
            DrawFootprintRange::new(DrawFootprintKind::GlobalWrite, address + 64, 64)
                .expect("writable consumer global footprint"),
        ]);
        assert_eq!(
            direct_draw_dependencies_are_gpu_orderable(
                41,
                &[fence],
                &checkpoints,
                &writable,
                |_, _| true,
                |_| true,
            )
            .map(|(admission, _)| admission),
            Err(SharedEventDrawReject::OverlapCopiedGlobal),
            "a writable consumer needs a new ordered copy and cannot alias the retained version"
        );
    }

    #[test]
    fn image_footprint_starts_at_the_selected_array_slice() {
        let mut space = GuestAddressSpace::new(64 * GUEST_PAGE_SIZE).expect("guest address space");
        let backing = GuestPhysicalBacking::new(20 * GUEST_PAGE_SIZE).expect("array backing");
        let address = space
            .map_shared(
                GuestMapPlacement::Anywhere,
                20 * GUEST_PAGE_SIZE,
                GUEST_PAGE_SIZE,
                GuestProtection::READ_WRITE,
                &backing,
                0,
            )
            .expect("array mapping")
            .get();
        let width = 64_u32;
        let height = 64_u32;
        let encoded_width = width - 1;
        let mut words = [0_u32; 8];
        words[0] = u32::try_from(address >> 8).expect("low address fits");
        words[1] = u32::try_from(address >> 40).expect("high address fits")
            | (130 << 20)
            | ((encoded_width & 0x3) << 30);
        words[2] = (encoded_width >> 2) | ((height - 1) << 14);
        words[3] = 0xfac | (27 << 20) | (13 << 28);
        words[4] = (3 << 16) | 4;
        let sampled = decode_image_descriptor_without_diagnostics(&words)
            .expect("nonzero-base array descriptor");
        let (_, byte_count) = sampled_texture_raw_range(sampled).expect("selected range");
        let mut ranges = Vec::new();
        append_image_range(
            &mut ranges,
            DrawFootprintKind::ImageRead,
            sampled,
            u64::try_from(byte_count).expect("byte count fits"),
        )
        .expect("image footprint");
        assert_eq!(ranges[0].guest_address, address + 3 * 65_536);
        assert_eq!(ranges[0].byte_count, 2 * 65_536);
    }

    #[test]
    fn same_queue_internal_resident_write_is_versioned_and_admits_the_sampling_draw() {
        // Plan A2: internal writes (mip promotion, stale-read rematerialisation) version their
        // resident on the queue timeline exactly like guest render-target writes, instead of
        // draining the host. A version recorded for a guest write must not satisfy an internal
        // producer range, and vice versa.
        let mut space = GuestAddressSpace::new(4 * GUEST_PAGE_SIZE).expect("guest address space");
        let backing = GuestPhysicalBacking::new(GUEST_PAGE_SIZE).expect("backing");
        let mapped = space
            .map_shared(
                GuestMapPlacement::Anywhere,
                GUEST_PAGE_SIZE,
                GUEST_PAGE_SIZE,
                GuestProtection::READ_WRITE,
                &backing,
                0,
            )
            .expect("mapping");
        let address = mapped.get();
        let sampled =
            decode_image_descriptor_without_diagnostics(&image_descriptor(address, 32, 32))
                .expect("sample descriptor");
        let (_, byte_count) = sampled_texture_raw_range(sampled).expect("sample range");
        let byte_count = u64::try_from(byte_count).expect("sample bytes fit u64");
        let span = ResourceSpan::new(address, byte_count);
        let producer = PhysicalFootprintRange::new(
            SharedEventWriteKind::InternalResidentDestination,
            address,
            byte_count,
            span.backing(),
        )
        .expect("internal producer footprint");
        let mut consumer =
            DrawFootprintRange::new(DrawFootprintKind::ImageRead, address, byte_count)
                .expect("sample consumer footprint");
        consumer.image = Some(sampled);
        let footprint = DirectDrawFootprint::from_ranges(vec![consumer]);

        let device = Device::system_default().expect("Metal device");
        let metal_descriptor = TextureDescriptor::new();
        metal_descriptor.set_texture_type(MTLTextureType::D2);
        metal_descriptor.set_pixel_format(sampled.guest.pixel_format);
        metal_descriptor.set_width(u64::from(sampled.guest.width));
        metal_descriptor.set_height(u64::from(sampled.guest.height));
        metal_descriptor.set_storage_mode(MTLStorageMode::Private);
        metal_descriptor.set_usage(MTLTextureUsage::RenderTarget | MTLTextureUsage::ShaderRead);
        let texture = device.new_texture(&metal_descriptor);
        let version = |kind| super::super::super::SharedEventResidentTextureVersion {
            texture: texture.clone(),
            descriptor: sampled.guest,
            texture_type: sampled.texture_type,
            layer_count: sampled.layer_count,
            mip_level_count: sampled.mip_level_count,
            content_version: 17,
            span,
            kind,
        };
        let producer_id = SharedEventProducerId::new(9);
        let fence = SharedEventFence::new(producer_id, 3);
        let checkpoint_with = |kind| SharedEventSignalCheckpoint {
            fence,
            global_writeback_count: 0,
            guest_write_range_count: 0,
            memory_write_count: 0,
            global_versions: Vec::new(),
            resident_texture_versions: vec![version(kind)],
            unresolved_write_ranges: Vec::new(),
            pending_write_footprint: vec![producer],
            footprint_complete: true,
        };
        let internal = HashMap::from([(
            producer_id,
            checkpoint_with(SharedEventWriteKind::InternalResidentDestination),
        )]);
        let (admission, overrides) = direct_draw_dependencies_are_gpu_orderable(
            9,
            &[fence],
            &internal,
            &footprint,
            |_, _| false,
            |_| true,
        )
        .expect("same queue internal resident version is GPU-orderable");
        assert_eq!(admission, DirectDrawAdmission::ResidentImage);
        assert_eq!(overrides.len(), 1);

        let mismatched = HashMap::from([(
            producer_id,
            checkpoint_with(SharedEventWriteKind::ResidentDestination),
        )]);
        assert_eq!(
            direct_draw_dependencies_are_gpu_orderable(
                9,
                &[fence],
                &mismatched,
                &footprint,
                |_, _| false,
                |_| true,
            )
            .map(|(admission, _)| admission),
            Err(SharedEventDrawReject::OverlapInternalResident),
            "a guest-write version must not license an internal producer range"
        );
    }

    #[test]
    fn same_queue_resident_image_retains_exact_texture_across_mapping_replacement() {
        let mut space = GuestAddressSpace::new(4 * GUEST_PAGE_SIZE).expect("guest address space");
        let original_backing = GuestPhysicalBacking::new(GUEST_PAGE_SIZE).expect("original");
        let replacement_backing = GuestPhysicalBacking::new(GUEST_PAGE_SIZE).expect("replacement");
        let mapped = space
            .map_shared(
                GuestMapPlacement::Anywhere,
                GUEST_PAGE_SIZE,
                GUEST_PAGE_SIZE,
                GuestProtection::READ_WRITE,
                &original_backing,
                0,
            )
            .expect("original mapping");
        let address = mapped.get();
        let sampled =
            decode_image_descriptor_without_diagnostics(&image_descriptor(address, 32, 32))
                .expect("sample descriptor");
        let (_, byte_count) = sampled_texture_raw_range(sampled).expect("sample range");
        let byte_count = u64::try_from(byte_count).expect("sample bytes fit u64");
        let span = ResourceSpan::new(address, byte_count);
        let producer = PhysicalFootprintRange::new(
            SharedEventWriteKind::ResidentDestination,
            address,
            byte_count,
            span.backing(),
        )
        .expect("resident producer footprint");
        let mut consumer =
            DrawFootprintRange::new(DrawFootprintKind::ImageRead, address, byte_count)
                .expect("sample consumer footprint");
        consumer.image = Some(sampled);
        let footprint = DirectDrawFootprint::from_ranges(vec![consumer]);

        let device = Device::system_default().expect("Metal device");
        let metal_descriptor = TextureDescriptor::new();
        metal_descriptor.set_texture_type(MTLTextureType::D2);
        metal_descriptor.set_pixel_format(sampled.guest.pixel_format);
        metal_descriptor.set_width(u64::from(sampled.guest.width));
        metal_descriptor.set_height(u64::from(sampled.guest.height));
        metal_descriptor.set_storage_mode(MTLStorageMode::Private);
        metal_descriptor.set_usage(MTLTextureUsage::RenderTarget | MTLTextureUsage::ShaderRead);
        let texture = device.new_texture(&metal_descriptor);
        let texture_pointer = texture.as_ptr();
        let producer_id = SharedEventProducerId::new(9);
        let fence = SharedEventFence::new(producer_id, 3);
        let checkpoint = SharedEventSignalCheckpoint {
            fence,
            global_writeback_count: 0,
            guest_write_range_count: 0,
            memory_write_count: 0,
            global_versions: Vec::new(),
            resident_texture_versions: vec![
                super::super::super::SharedEventResidentTextureVersion {
                    texture,
                    descriptor: sampled.guest,
                    texture_type: sampled.texture_type,
                    layer_count: sampled.layer_count,
                    mip_level_count: sampled.mip_level_count,
                    content_version: 17,
                    span,
                    kind: SharedEventWriteKind::ResidentDestination,
                },
            ],
            unresolved_write_ranges: Vec::new(),
            pending_write_footprint: vec![producer],
            footprint_complete: true,
        };
        let checkpoints = HashMap::from([(producer_id, checkpoint)]);
        let (admission, overrides) = direct_draw_dependencies_are_gpu_orderable(
            9,
            &[fence],
            &checkpoints,
            &footprint,
            |_, _| false,
            |_| true,
        )
        .expect("same queue exact resident version is GPU-orderable");
        assert_eq!(admission, DirectDrawAdmission::ResidentImage);
        assert_eq!(overrides.len(), 1);

        space
            .unmap(mapped, GUEST_PAGE_SIZE)
            .expect("unmap original");
        space
            .map_shared(
                GuestMapPlacement::Fixed(mapped),
                GUEST_PAGE_SIZE,
                GUEST_PAGE_SIZE,
                GuestProtection::READ_WRITE,
                &replacement_backing,
                0,
            )
            .expect("replacement mapping");
        assert_eq!(overrides[0].texture.as_ptr(), texture_pointer);
        assert_eq!(overrides[0].sampled, sampled);
        assert_eq!(
            direct_draw_dependencies_are_gpu_orderable(
                10,
                &[fence],
                &checkpoints,
                &footprint,
                |_, _| false,
                |_| true,
            )
            .map(|(admission, _)| admission),
            Err(SharedEventDrawReject::OverlapResident),
            "cross-queue resident images require general immutable ownership"
        );
    }

    fn current_resident_version(
        presenter: &MetalPresenter,
        descriptor: super::super::super::GuestTextureDescriptor,
    ) -> super::super::super::SharedEventResidentTextureVersion {
        let resident = presenter
            .resident_guest_textures
            .get(&descriptor)
            .expect("resident texture exists");
        super::super::super::SharedEventResidentTextureVersion {
            texture: resident.texture.clone(),
            descriptor: resident.descriptor,
            texture_type: resident.texture_type,
            layer_count: resident.layer_count,
            mip_level_count: resident.mip_level_count,
            content_version: resident.content_version,
            span: resident.span.expect("resident has physical span"),
            kind: SharedEventWriteKind::ResidentDestination,
        }
    }

    #[test]
    fn resident_version_preflight_rejects_later_gpu_cpu_and_reprotect_changes() {
        let mut memory = GuestAddressSpace::new(8 * GUEST_PAGE_SIZE).expect("guest address space");
        let backing = GuestPhysicalBacking::new(GUEST_PAGE_SIZE).expect("guest backing");
        let primary = memory
            .map_shared(
                GuestMapPlacement::Anywhere,
                GUEST_PAGE_SIZE,
                GUEST_PAGE_SIZE,
                GuestProtection::READ_WRITE,
                &backing,
                0,
            )
            .expect("primary mapping");
        let descriptor = super::super::super::GuestTextureDescriptor {
            address: primary.get(),
            width: 32,
            height: 32,
            pitch: 32,
            tile_mode: 0,
            pixel_format: metal::MTLPixelFormat::RGBA8Unorm,
            bytes_per_pixel: 4,
        };
        let (_sender, receiver) = std::sync::mpsc::channel();
        let mut presenter = MetalPresenter::new(receiver).expect("Metal presenter");
        let device = presenter.device.as_ref().expect("Metal device").clone();
        presenter
            .ensure_resident_texture(&device, descriptor)
            .expect("resident texture");
        presenter.mark_resident_texture_written(descriptor);
        let first = current_resident_version(&presenter, descriptor);
        assert!(retained_resident_version_is_current(
            &presenter.resident_guest_textures,
            &first,
        ));

        presenter.mark_resident_texture_written(descriptor);
        assert!(
            !retained_resident_version_is_current(&presenter.resident_guest_textures, &first),
            "a later encode can destructively update the same Metal object"
        );
        let second = current_resident_version(&presenter, descriptor);
        assert!(retained_resident_version_is_current(
            &presenter.resident_guest_textures,
            &second,
        ));
        memory
            .write(primary, &[0x5a; 32])
            .expect("same-backing CPU write");
        assert!(
            !retained_resident_version_is_current(&presenter.resident_guest_textures, &second),
            "same-backing CPU dirtiness obsoletes admission"
        );

        let second_backing = GuestPhysicalBacking::new(GUEST_PAGE_SIZE).expect("second backing");
        let second_primary = memory
            .map_shared(
                GuestMapPlacement::Anywhere,
                GUEST_PAGE_SIZE,
                GUEST_PAGE_SIZE,
                GuestProtection::READ_WRITE,
                &second_backing,
                0,
            )
            .expect("second primary mapping");
        let second_descriptor = super::super::super::GuestTextureDescriptor {
            address: second_primary.get(),
            ..descriptor
        };
        presenter
            .ensure_resident_texture(&device, second_descriptor)
            .expect("second resident texture");
        presenter.mark_resident_texture_written(second_descriptor);
        let before_reprotect = current_resident_version(&presenter, second_descriptor);
        assert!(retained_resident_version_is_current(
            &presenter.resident_guest_textures,
            &before_reprotect,
        ));
        memory
            .map_shared(
                GuestMapPlacement::Anywhere,
                GUEST_PAGE_SIZE,
                GUEST_PAGE_SIZE,
                GuestProtection::READ_WRITE,
                &second_backing,
                0,
            )
            .expect("late coherent alias");
        assert!(
            !retained_resident_version_is_current(
                &presenter.resident_guest_textures,
                &before_reprotect,
            ),
            "same-physical-backing watch reprotection rejects the old token"
        );
    }
}
