// Copyright (C) 2026 REmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

//! Pure consumer-footprint classification for direct draws behind exact `SharedEvent` waits.
//!
//! This module deliberately does not make draws GPU-orderable. It proves which blocked draws
//! would be physically disjoint from every exact producer checkpoint and reports the rejection
//! histogram. Admission remains default-deny until live measurements and scheduler integration
//! tests establish that the candidates also avoid internal Metal drains.

use super::super::{CB_COLOR_CONTROL, ColorOperationMode, ResourceSpan};
use super::super::{
    DEFAULT_CB_COLOR_CONTROL, GuestQueueBatchState, MetalPresenter, PhysicalFootprintRange,
    SharedEventFence, SharedEventProducerId, SharedEventSignalCheckpoint, SharedEventWriteKind,
    color_operation_is_raster_noop, decode_color_operation_mode,
    decode_depth_target_without_diagnostics, decode_image_descriptor_without_diagnostics,
    decode_render_targets, draw_diagnostics_are_host_effectful, effective_global_memory_binding,
    image_resource_descriptor_is_null, register_value, resident_texture_raw_byte_count,
    sampled_texture_raw_range,
};
use remu_gpu::{IndexedDraw, ShaderSnapshot};
use remu_memory::GuestSharedMappingIdentity;
use remu_shader::{InstructionControl, ShaderEncoding, ShaderEvaluation};
use std::collections::HashMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

pub(crate) const SHARED_EVENT_DRAW_REJECT_COUNT: usize = 8;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(usize)]
pub(crate) enum SharedEventDrawReject {
    Overlap,
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
            Self::Overlap => "overlap",
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
    Global,
    ScalarEvaluationRead,
    Image,
}

impl DrawFootprintKind {
    const COUNT: usize = 5;

    const fn index(self) -> usize {
        self as usize
    }

    const fn label(self) -> &'static str {
        match self {
            Self::ColorAttachment => "color",
            Self::DepthAttachment => "depth",
            Self::Global => "global",
            Self::ScalarEvaluationRead => "scalar-eval",
            Self::Image => "image",
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct DrawFootprintRange {
    kind: DrawFootprintKind,
    guest_address: u64,
    byte_count: u64,
    mapping: GuestSharedMappingIdentity,
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

fn program_memory_is_completely_covered(
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
        append_range(
            ranges,
            DrawFootprintKind::Global,
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
        append_range(
            ranges,
            DrawFootprintKind::Image,
            descriptor.guest.address,
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

fn direct_draw_dependencies_are_disjoint(
    dependencies: &[SharedEventFence],
    checkpoints: &HashMap<SharedEventProducerId, SharedEventSignalCheckpoint>,
    footprint: &DirectDrawFootprint,
) -> Result<(), SharedEventDrawReject> {
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
        if checkpoint.pending_write_footprint.iter().any(|&producer| {
            footprint
                .ranges
                .iter()
                .any(|&consumer| consumer.overlaps_producer(producer))
        }) {
            return Err(SharedEventDrawReject::Overlap);
        }
    }
    Ok(())
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
        DrawFootprintKind::Global,
        DrawFootprintKind::ScalarEvaluationRead,
        DrawFootprintKind::Image,
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
    ) -> bool {
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
                    direct_draw_dependencies_are_disjoint(
                        dependencies,
                        &state.resource_dependencies,
                        footprint,
                    )
                }),
        };
        match decision {
            Ok(()) => {
                self.statistics.shared_event_draw_disjoint_candidates = self
                    .statistics
                    .shared_event_draw_disjoint_candidates
                    .saturating_add(1);
            }
            Err(reject) => {
                self.statistics.shared_event_draw_rejections[reject.index()] =
                    self.statistics.shared_event_draw_rejections[reject.index()].saturating_add(1);
            }
        }

        if std::env::var_os("REMU_TRACE_SHARED_EVENT_DRAWS").is_some_and(|value| value == "1") {
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
                        .map_or_else(|reject| reject.label(), |()| "disjoint"),
                    footprint
                        .as_ref()
                        .map_or(0, DirectDrawFootprint::byte_count),
                );
            }
        }
        decision.is_ok()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use remu_gpu::{GpuRegister, ShaderSnapshot};
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
        registers: &mut Vec<GpuRegister>,
        slot: u32,
        address: u64,
        width: u32,
        height: u32,
    ) {
        let primary = slot * super::super::super::CB_COLOR_REGISTER_STRIDE;
        registers.extend([
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
        ]);
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
            index_snapshot: Some(Arc::from([0, 1, 2])),
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
            context_registers: Vec::new(),
            shader_registers: Vec::new(),
            user_config_registers: Vec::new(),
        };
        append_color_target(&mut draw.context_registers, 0, base, 64, 64);
        append_color_target(
            &mut draw.context_registers,
            1,
            base + GUEST_PAGE_SIZE,
            64,
            64,
        );
        draw.context_registers.push(GpuRegister {
            offset: super::super::super::CB_TARGET_MASK,
            value: 0xff,
        });

        let depth = base + 2 * GUEST_PAGE_SIZE;
        let depth_base =
            u32::try_from(depth >> 8).expect("test depth-target address low bits fit u32");
        let depth_base_hi = ((depth >> 40) & 0xff) as u32;
        draw.context_registers.extend([
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
        ]);
        draw.context_registers
            .sort_unstable_by_key(|register| register.offset);

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
        export_evaluation.image_bindings.push(ImageBinding {
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
        pixel_evaluation.image_bindings.push(ImageBinding {
            pc: 20,
            opcode: Opcode::Known("ImageStore"),
            control: image_control(),
            resource_descriptor: image_descriptor(pixel_image, 32, 32).into(),
            sampler_descriptor: Vec::new().into(),
            mip_level: None,
        });
        draw.export_shader_evaluation = Some(Arc::new(export_evaluation));
        draw.pixel_shader_evaluation = Some(Arc::new(pixel_evaluation));
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
            DrawFootprintKind::Global,
            DrawFootprintKind::ScalarEvaluationRead,
            DrawFootprintKind::Image,
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
    fn fragmentless_depth_only_draw_is_structurally_complete() {
        let (_space, base) = mapped_space(16);
        let mut draw = complete_draw(base);
        draw.context_registers.retain(|register| {
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
        Arc::make_mut(
            missing_image
                .export_shader_evaluation
                .as_mut()
                .expect("export evaluation"),
        )
        .image_bindings
        .clear();
        cases.push((missing_image, SharedEventDrawReject::ShaderCoverage));

        let mut missing_export = draw.clone();
        missing_export.export_shader_evaluation = None;
        cases.push((missing_export, SharedEventDrawReject::ShaderCoverage));

        let mut missing_pixel = draw.clone();
        missing_pixel.pixel_shader_evaluation = None;
        cases.push((missing_pixel, SharedEventDrawReject::ShaderCoverage));

        for operation in [2, 3, 5] {
            let mut special = draw.clone();
            special.context_registers.push(GpuRegister {
                offset: super::super::super::CB_COLOR_CONTROL,
                value: operation << 4,
            });
            special
                .context_registers
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
        let evaluation = Arc::make_mut(
            draw.export_shader_evaluation
                .as_mut()
                .expect("export evaluation"),
        );
        evaluation.global_memory_bindings.push(GlobalMemoryBinding {
            scalar_address: 99,
            base_address: 0,
            instruction_pcs: Vec::new().into(),
            byte_count: 4,
            data: vec![0; 4],
            writable: false,
            write_back_to_guest: false,
        });
        evaluation.image_bindings.push(ImageBinding {
            pc: 999,
            opcode: Opcode::Known("ImageSample"),
            control: image_control(),
            resource_descriptor: vec![0; 8].into(),
            sampler_descriptor: vec![0; 4].into(),
            mip_level: None,
        });
        direct_draw_footprint(&draw, false).expect("true null stubs add no guest interval");

        let evaluation = Arc::make_mut(
            draw.export_shader_evaluation
                .as_mut()
                .expect("export evaluation"),
        );
        evaluation.global_memory_bindings.push(GlobalMemoryBinding {
            scalar_address: 100,
            base_address: 0x7fff_0000_0000,
            instruction_pcs: Vec::new().into(),
            byte_count: 16,
            data: vec![0; 16],
            writable: false,
            write_back_to_guest: false,
        });
        assert_eq!(
            direct_draw_footprint(&draw, false),
            Err(SharedEventDrawReject::Unresolved)
        );

        let mut malformed_image = complete_draw(base);
        Arc::make_mut(
            malformed_image
                .export_shader_evaluation
                .as_mut()
                .expect("export evaluation"),
        )
        .image_bindings[0]
            .resource_descriptor = vec![1].into();
        assert_eq!(
            direct_draw_footprint(&malformed_image, false),
            Err(SharedEventDrawReject::Unresolved)
        );

        let mut overflowing_scalar = complete_draw(base);
        Arc::make_mut(
            overflowing_scalar
                .export_shader_evaluation
                .as_mut()
                .expect("export evaluation"),
        )
        .scalar_memory_reads
        .push((u64::MAX - 1, 0));
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
                let evaluation = Arc::make_mut(
                    draw.export_shader_evaluation
                        .as_mut()
                        .expect("export evaluation"),
                );
                evaluation
                    .global_memory_bindings
                    .iter_mut()
                    .for_each(|binding| binding.instruction_pcs.retain(|covered| *covered != pc));
                evaluation.vertex_inputs.retain(|binding| binding.pc != pc);
                evaluation.image_bindings.retain(|binding| binding.pc != pc);
            } else {
                let evaluation = Arc::make_mut(
                    draw.pixel_shader_evaluation
                        .as_mut()
                        .expect("pixel evaluation"),
                );
                evaluation
                    .global_memory_bindings
                    .iter_mut()
                    .for_each(|binding| binding.instruction_pcs.retain(|covered| *covered != pc));
                evaluation.image_bindings.retain(|binding| binding.pc != pc);
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
            DrawFootprintRange::new(DrawFootprintKind::Global, second + 64, 64)
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
            unresolved_write_ranges: Vec::new(),
            pending_write_footprint: vec![producer],
            footprint_complete: true,
        };
        let mut checkpoints = HashMap::from([(producer_id, checkpoint.clone())]);
        assert_eq!(
            direct_draw_dependencies_are_disjoint(&[fence], &checkpoints, &consumer),
            Err(SharedEventDrawReject::Overlap)
        );

        checkpoints.get_mut(&producer_id).expect("checkpoint").fence =
            SharedEventFence::new(producer_id, 8);
        assert_eq!(
            direct_draw_dependencies_are_disjoint(&[fence], &checkpoints, &consumer),
            Err(SharedEventDrawReject::IncompleteProducer)
        );
        checkpoints.get_mut(&producer_id).expect("checkpoint").fence = fence;
        checkpoints
            .get_mut(&producer_id)
            .expect("checkpoint")
            .footprint_complete = false;
        assert_eq!(
            direct_draw_dependencies_are_disjoint(&[fence], &checkpoints, &consumer),
            Err(SharedEventDrawReject::IncompleteProducer)
        );

        let rebound_consumer = DirectDrawFootprint::from_ranges(vec![DrawFootprintRange {
            kind: DrawFootprintKind::Global,
            guest_address: first + 32,
            byte_count: 64,
            mapping: GuestSharedMappingIdentity {
                backing_id: producer.mapping.backing_id + 1,
                backing_offset: producer.mapping.backing_offset,
            },
        }]);
        let exact_checkpoint = SharedEventSignalCheckpoint {
            footprint_complete: true,
            ..checkpoint
        };
        let rebound_checkpoints = HashMap::from([(producer_id, exact_checkpoint)]);
        assert_eq!(
            direct_draw_dependencies_are_disjoint(
                &[fence],
                &rebound_checkpoints,
                &rebound_consumer,
            ),
            Ok(()),
            "identical virtual addresses rebound to different captured backings are disjoint"
        );
    }
}
