// Copyright (C) 2026 REmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

use super::super::completion::{
    COMMAND_BUFFER_SETTLE_TIMEOUT, command_buffer_failure_message, settle_timeout_message,
    wait_until_settled_within,
};
use super::super::shared_event::{
    SharedEventBatchTerminalState, SharedEventFence, SharedEventProducerId,
    terminal_shared_event_advance,
};
use super::super::{
    FlushPolicy, GuestQueueBatchState, PendingSharedEventAckTrace, PresenterStatistics,
    SharedEventAckNativeHostTimes, SharedEventAckTerminalTimes, SharedEventAckTrace,
    TerminalPublicationPhase, TerminalPublicationTimer,
};
use super::{
    GLOBAL_UPLOAD_ALIGNMENT, GLOBAL_UPLOAD_CHUNK_BYTE_COUNT, GlobalUploadChunk,
    GuestBatchFlushReason, GuestBatchWork, GuestCommandBatch, MetalPresenter, OrderedEventSource,
    PendingGlobalWriteback, PendingGuestWriteRange, UploadedGlobalBuffer,
    guest_physical_ranges_overlap, guest_ranges_overlap,
};
use crate::presenter::diagnostics::journal_guest_write;
use crate::presenter::{
    PhysicalFootprintRange, SharedEventGlobalBufferVersion, SharedEventResidentTextureVersion,
    SharedEventSignalCheckpoint, SharedEventUnresolvedWriteRange, SharedEventWriteKind,
};
use block::ConcreteBlock;
use metal::foreign_types::ForeignType;
use metal::{
    Buffer, CommandBuffer, ComputeCommandEncoder, Device, MTLCommandBufferStatus,
    MTLResourceOptions,
};
use objc::{msg_send, sel, sel_impl};
use remu_memory::{
    CpuWriteWatch, GuestRangeLease, GuestSharedMappingIdentity, GuestWriteSite,
    LeasePublishOutcome, guest_shared_backing_identity, read_guest_shared_backing,
};
use remu_shader::ShaderEvaluation;
use std::io::ErrorKind;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

static DISCARDED_STALE_GLOBAL_WRITEBACKS: AtomicUsize = AtomicUsize::new(0);
static KEPT_GUEST_BYTES_WRITEBACK_RUNS: AtomicUsize = AtomicUsize::new(0);

/// Reports a changed run left untouched because the guest wrote inside it after the capture.
///
/// This is the loud form of the continuity guard: a writeback that yields to newer guest bytes
/// trades corruption for at-worst stale GPU output, and the trade must stay visible.
fn trace_kept_guest_bytes_over_writeback(address: u64, byte_count: usize, mode: &str) {
    let sequence = KEPT_GUEST_BYTES_WRITEBACK_RUNS.fetch_add(1, Ordering::Relaxed) + 1;
    if sequence <= 16 || sequence.is_power_of_two() {
        eprintln!(
            "[REMu][METAL] kept guest bytes over a writable global writeback run sequence={sequence} mode={mode} address={address:#x} bytes={byte_count}: the guest wrote inside the range after capture"
        );
    }
}

#[derive(Debug, PartialEq, Eq)]
enum GuestBatchSettlement {
    Completed,
    Failed(String),
    Pending(String),
}

/// Outcome of settling the oldest committed batch. `Retired` carries that batch's own completion
/// result; `Pending` means it is still queued and no slot was reclaimed.
enum FrontBatchSettlement {
    Empty,
    Retired(Result<(), String>),
    Pending(String),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum GlobalWritebackOutcome {
    Applied,
    DiscardedStale,
}

impl SharedEventSignalCheckpoint {
    pub(crate) fn capture_queue_prefix(
        state: &GuestQueueBatchState,
        fence: SharedEventFence,
    ) -> Self {
        let batch = state
            .open_batch
            .as_ref()
            .expect("a queue signal has an open batch");
        let mut checkpoint = Self {
            fence,
            global_writeback_count: batch.global_writebacks.len(),
            guest_write_range_count: batch.guest_write_ranges.len(),
            memory_write_count: batch.memory_writes.len(),
            global_versions: Vec::new(),
            resident_texture_versions: Vec::new(),
            unresolved_write_ranges: Vec::new(),
            pending_write_footprint: Vec::new(),
            footprint_complete: true,
        };
        for earlier in &state.committed_batches {
            checkpoint.capture_batch_prefix(
                earlier,
                earlier.global_writebacks.len(),
                earlier.guest_write_ranges.len(),
                earlier.memory_writes.len(),
            );
        }
        checkpoint.capture_batch_prefix(
            batch,
            checkpoint.global_writeback_count,
            checkpoint.guest_write_range_count,
            checkpoint.memory_write_count,
        );
        checkpoint
    }

    fn capture_batch_prefix(
        &mut self,
        batch: &GuestCommandBatch,
        global_writeback_count: usize,
        guest_write_range_count: usize,
        memory_write_count: usize,
    ) {
        self.footprint_complete &= batch.resident_write_footprint_complete;
        for &range in &batch.resident_write_footprint {
            append_unique_physical_footprint(&mut self.pending_write_footprint, range);
        }
        for version in &batch.resident_texture_versions {
            if let Some(existing) = self.resident_texture_versions.iter_mut().find(|existing| {
                existing.span.backing() == version.span.backing()
                    && existing.span.len() == version.span.len()
            }) {
                *existing = version.clone();
            } else {
                self.resident_texture_versions.push(version.clone());
            }
        }
        for version in &batch.global_buffer_versions {
            self.capture_global_version(version.clone());
        }
        for writeback in &batch.global_writebacks[..global_writeback_count] {
            let Ok(byte_count) = u64::try_from(writeback.byte_count) else {
                self.footprint_complete = false;
                continue;
            };
            let version = SharedEventGlobalBufferVersion {
                buffer: writeback.buffer.clone(),
                offset: writeback.offset,
                guest_address: writeback.guest_address,
                byte_count,
                mapping_identity: writeback.mapping_identity(),
            };
            self.capture_global_version(version);
        }
        for range in &batch.guest_write_ranges[..guest_write_range_count] {
            self.unresolved_write_ranges
                .push(SharedEventUnresolvedWriteRange {
                    guest_address: range.address,
                    byte_count: range.byte_count,
                    mapping_identity: Some(range.physical_identity),
                });
            self.capture_physical_range(
                SharedEventWriteKind::ZeroCopyGlobal,
                range.address,
                range.byte_count,
                Some(range.physical_identity),
            );
        }
        for write in &batch.memory_writes[..memory_write_count] {
            let (guest_address, byte_count) = write.range();
            self.unresolved_write_ranges
                .push(SharedEventUnresolvedWriteRange {
                    guest_address,
                    byte_count,
                    mapping_identity: write.mapping_identity,
                });
            self.capture_physical_range(
                SharedEventWriteKind::TerminalGuestWrite,
                guest_address,
                byte_count,
                write.mapping_identity,
            );
        }
    }

    fn capture_global_version(&mut self, version: SharedEventGlobalBufferVersion) {
        // Each writable copied binding gets fresh Metal storage. Replacing only the same exact
        // guest-physical interval materializes the latest version as of this signal; partially
        // overlapping intervals remain separate so consumers reject ambiguity.
        if let Some(existing) = self.global_versions.iter_mut().find(|existing| {
            existing.byte_count == version.byte_count
                && match (existing.mapping_identity, version.mapping_identity) {
                    (Some(existing), Some(version)) => existing == version,
                    (None, None) => existing.guest_address == version.guest_address,
                    _ => false,
                }
        }) {
            *existing = version.clone();
        } else {
            self.global_versions.push(version.clone());
        }
        self.capture_physical_range(
            SharedEventWriteKind::CopiedGlobal,
            version.guest_address,
            version.byte_count,
            version.mapping_identity,
        );
    }

    fn capture_physical_range(
        &mut self,
        kind: SharedEventWriteKind,
        guest_address: u64,
        byte_count: u64,
        mapping: Option<GuestSharedMappingIdentity>,
    ) {
        match PhysicalFootprintRange::new(kind, guest_address, byte_count, mapping) {
            Some(range) => {
                append_unique_physical_footprint(&mut self.pending_write_footprint, range);
            }
            None => self.footprint_complete = false,
        }
    }
}

impl PhysicalFootprintRange {
    pub(crate) fn new(
        kind: SharedEventWriteKind,
        guest_address: u64,
        byte_count: u64,
        mapping: Option<GuestSharedMappingIdentity>,
    ) -> Option<Self> {
        let mapping = mapping?;
        if byte_count == 0 || guest_address.checked_add(byte_count).is_none() {
            return None;
        }
        mapping.backing_offset.checked_add(byte_count)?;
        Some(Self {
            kind,
            guest_address,
            byte_count,
            mapping,
        })
    }

    // This is the canonical physical comparator for the next disjoint-draw slice. The producer
    // journal lands first so no draw is whitelisted against an incomplete footprint.
    #[allow(dead_code)]
    pub(crate) fn overlaps(self, other: Self) -> bool {
        if self.mapping.backing_id != other.mapping.backing_id {
            return false;
        }
        let Some(self_end) = self.mapping.backing_offset.checked_add(self.byte_count) else {
            return true;
        };
        let Some(other_end) = other.mapping.backing_offset.checked_add(other.byte_count) else {
            return true;
        };
        self.mapping.backing_offset < other_end && other.mapping.backing_offset < self_end
    }
}

fn append_unique_physical_footprint(
    destination: &mut Vec<PhysicalFootprintRange>,
    range: PhysicalFootprintRange,
) {
    if !destination.contains(&range) {
        destination.push(range);
    }
}

impl GuestCommandBatch {
    pub(crate) fn record_resident_texture_version(
        &mut self,
        version: SharedEventResidentTextureVersion,
    ) {
        if let Some(existing) = self.resident_texture_versions.iter_mut().find(|existing| {
            existing.kind == version.kind
                && existing.span.backing() == version.span.backing()
                && existing.span.len() == version.span.len()
        }) {
            *existing = version;
        } else {
            self.resident_texture_versions.push(version);
        }
    }

    pub(crate) fn record_resident_write_footprint(
        &mut self,
        kind: SharedEventWriteKind,
        span: Option<super::super::ResourceSpan>,
    ) {
        let Some(span) = span else {
            self.resident_write_footprint_complete = false;
            return;
        };
        let Some(range) =
            PhysicalFootprintRange::new(kind, span.address(), span.len(), span.backing())
        else {
            self.resident_write_footprint_complete = false;
            return;
        };
        append_unique_physical_footprint(&mut self.resident_write_footprint, range);
    }
}

impl MetalPresenter {
    pub(crate) fn shared_event_ack_phase_timing_enabled(&self) -> bool {
        self.shared_event_ack_phase_timing
    }

    pub(crate) fn record_shared_event_ack_batch_cpu_phases(
        &mut self,
        translation: Duration,
        metal_encoding: Duration,
        compute: bool,
    ) {
        if !self.shared_event_ack_phase_timing {
            return;
        }
        let Some(queue_id) = self.active_guest_queue else {
            return;
        };
        let phases = self.shared_event_ack_batch_cpu.entry(queue_id).or_default();
        phases.translation = phases.translation.saturating_add(translation);
        phases.metal_encoding = phases.metal_encoding.saturating_add(metal_encoding);
        if compute {
            phases.computes = phases.computes.saturating_add(1);
        } else {
            phases.draws = phases.draws.saturating_add(1);
        }
    }

    pub(crate) fn prepare_shared_event_ack_phase_trace(
        &mut self,
        fence: Option<SharedEventFence>,
        event_queue: Duration,
        host_preflight: Duration,
        ack_encoding: Duration,
    ) {
        if !self.shared_event_ack_phase_timing
            || self.shared_event_ack_phase_samples_remaining == 0
            || self.shared_event_ack_traces.len() + self.pending_shared_event_ack_traces.len() >= 64
        {
            return;
        }
        let (Some(queue_id), Some(fence)) = (self.active_guest_queue, fence) else {
            return;
        };
        self.shared_event_ack_phase_samples_remaining -= 1;
        self.pending_shared_event_ack_traces.insert(
            queue_id,
            PendingSharedEventAckTrace {
                fence,
                event_queue,
                host_preflight,
                ack_encoding,
            },
        );
    }

    pub(crate) fn record_shared_event_ack_observed(
        &mut self,
        fence: SharedEventFence,
        observed_at: Instant,
    ) {
        let Some(&command_buffer_key) = self.shared_event_ack_trace_buffers_by_fence.get(&fence)
        else {
            return;
        };
        if let Some(trace) = self.shared_event_ack_traces.get_mut(&command_buffer_key) {
            trace.ack_observed_at = Some(observed_at);
        }
        self.finish_shared_event_ack_trace(command_buffer_key);
    }

    fn record_shared_event_ack_terminal(&mut self, command_buffer: &CommandBuffer) {
        let command_buffer_key = command_buffer.as_ptr() as usize;
        let Some(trace) = self.shared_event_ack_traces.get_mut(&command_buffer_key) else {
            return;
        };
        trace.terminal = Some(completed_command_buffer_ack_timeline(command_buffer));
        self.finish_shared_event_ack_trace(command_buffer_key);
    }

    fn finish_shared_event_ack_trace(&mut self, command_buffer_key: usize) {
        let ready = self
            .shared_event_ack_traces
            .get(&command_buffer_key)
            .is_some_and(|trace| trace.terminal.is_some() && trace.ack_observed_at.is_some());
        if !ready {
            return;
        }
        let trace = self
            .shared_event_ack_traces
            .remove(&command_buffer_key)
            .expect("ready acknowledgement trace exists");
        self.shared_event_ack_trace_buffers_by_fence
            .remove(&trace.fence);
        let terminal = trace.terminal.expect("ready trace has terminal timing");
        let native_host = trace
            .native_host
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let statistics = &mut self.statistics.shared_event_ack_phases;
        statistics.samples = statistics.samples.saturating_add(1);
        statistics.draws = statistics.draws.saturating_add(trace.cpu.draws);
        statistics.computes = statistics.computes.saturating_add(trace.cpu.computes);
        statistics.event_queue.record(trace.event_queue);
        statistics.cpu_translation.record(trace.cpu.translation);
        statistics.metal_encoding.record(trace.cpu.metal_encoding);
        statistics.ack_encoding.record(trace.ack_encoding);
        statistics.commit_call.record(trace.commit_call);
        if let Some(scheduled_at) = native_host.scheduled_at {
            statistics
                .commit_to_scheduled
                .record(scheduled_at.saturating_duration_since(trace.committed_at));
            if let Some(completed_at) = native_host.completed_at {
                statistics
                    .scheduled_to_completed
                    .record(completed_at.saturating_duration_since(scheduled_at));
            }
        }
        statistics
            .kernel_to_gpu_start
            .record(terminal.kernel_to_gpu_start);
        statistics.gpu_execution.record(terminal.gpu_execution);
        statistics
            .gpu_to_kernel_end
            .record(terminal.gpu_to_kernel_end);
        statistics.commit_to_ack.record(
            trace
                .ack_observed_at
                .expect("ready trace has acknowledgement observation")
                .saturating_duration_since(trace.committed_at),
        );
    }

    pub(crate) fn encode_active_guest_queue_host_ack(
        &mut self,
    ) -> Result<Option<super::super::SharedEventFence>, String> {
        let queue_id = self
            .active_guest_queue
            .ok_or("shared-event host acknowledgement has no active guest queue")?;
        self.ensure_active_open_guest_batch()?;
        let state = self
            .guest_queue_batches
            .get_mut(&queue_id)
            .ok_or_else(|| format!("missing host-ack guest queue {queue_id}"))?;
        let fence = match state.shared_event_timeline.allocate_fence() {
            Ok(fence) => fence,
            Err(_) => {
                self.statistics.shared_event_fallbacks_timeline_overflow = self
                    .statistics
                    .shared_event_fallbacks_timeline_overflow
                    .saturating_add(1);
                return Ok(None);
            }
        };
        {
            let batch = state
                .open_batch
                .as_mut()
                .expect("host acknowledgement opened a consumer batch");
            batch.close_active_pass_encoders();
            batch
                .command_buffer
                .encode_signal_event(state.shared_event_timeline.event(), fence.value());
            batch.max_encoded_signal = Some(fence);
        }
        let checkpoint = SharedEventSignalCheckpoint::capture_queue_prefix(state, fence);
        if !checkpoint.footprint_complete {
            self.statistics.shared_event_incomplete_signal_footprints = self
                .statistics
                .shared_event_incomplete_signal_footprints
                .saturating_add(1);
        }
        debug_assert!(
            checkpoint.footprint_complete,
            "shared-event host acknowledgement has an incomplete producer footprint"
        );
        state
            .open_batch
            .as_mut()
            .expect("host acknowledgement batch remains open")
            .signal_checkpoints
            .push(checkpoint);
        self.statistics.shared_event_signals =
            self.statistics.shared_event_signals.saturating_add(1);
        Ok(Some(fence))
    }

    pub(crate) fn unsignaled_host_dependencies(
        &mut self,
        queue_id: u64,
    ) -> Vec<super::super::SharedEventFence> {
        let host_retired = self
            .guest_queue_batches
            .iter()
            .map(|(&producer_queue_id, state)| {
                (
                    SharedEventProducerId::new(producer_queue_id),
                    state.host_retired_fence_value,
                )
            })
            .collect::<std::collections::HashMap<_, _>>();
        let Some(consumer) = self.guest_queue_batches.get_mut(&queue_id) else {
            return Vec::new();
        };
        let before = consumer.host_dependencies.len();
        consumer.host_dependencies.retain(|producer, value| {
            host_retired.get(producer).copied().unwrap_or_default() < *value
        });
        consumer
            .resource_dependencies
            .retain(|producer, checkpoint| {
                host_retired.get(producer).copied().unwrap_or_default() < checkpoint.fence.value()
            });
        if consumer.host_dependencies.len() < before {
            self.statistics.shared_event_host_wakes =
                self.statistics.shared_event_host_wakes.saturating_add(1);
        }

        consumer
            .host_dependencies
            .iter()
            .map(|(&producer, &value)| super::super::SharedEventFence::new(producer, value))
            .collect()
    }

    /// Selects the command queue and ordering for a translated scanout snapshot.
    ///
    /// A queue-sourced flip snapshots on its producing queue and publishes a fence because
    /// presentation and surface-less retirement run on the legacy host queue. A host-sourced
    /// flip (interactive tick, video-out service) has no producing queue: it snapshots on the
    /// legacy host queue itself, ordered behind every guest queue's committed work by waiting
    /// on each timeline at its newest committed fence value. Only committed batches have their
    /// signals encoded, so these waits cannot wedge the queue.
    pub(crate) fn scanout_snapshot_ordering(
        &mut self,
    ) -> Result<
        (
            metal::CommandQueue,
            Option<super::super::SharedEventFence>,
            Vec<(metal::SharedEvent, u64)>,
        ),
        String,
    > {
        if !self.per_queue_guest_batches || self.active_guest_queue.is_some() {
            let queue = self.active_guest_command_queue()?;
            self.flush_guest_command_batch(
                FlushPolicy::Required,
                GuestBatchFlushReason::FlipSnapshot,
            )?;
            let snapshot_ready = if self.per_queue_guest_batches {
                let queue_id = self
                    .active_guest_queue
                    .expect("per-queue snapshot ordering checked its active queue");
                let state = self.guest_queue_batches.get_mut(&queue_id).ok_or_else(|| {
                    format!("translated scanout guest queue {queue_id} is unavailable")
                })?;
                Some(
                    state
                        .shared_event_timeline
                        .allocate_fence()
                        .map_err(|error| error.to_string())?,
                )
            } else {
                None
            };
            return Ok((queue, snapshot_ready, Vec::new()));
        }
        self.flush_all_guest_command_batches_by(
            FlushPolicy::Required,
            GuestBatchFlushReason::FlipSnapshot,
            Instant::now() + COMMAND_BUFFER_SETTLE_TIMEOUT,
        )?;
        let cross_queue_waits = self
            .guest_queue_batches
            .values()
            .filter_map(|state| {
                let newest = state
                    .committed_batches
                    .iter()
                    .filter_map(|batch| batch.max_encoded_signal)
                    .map(|fence| fence.value())
                    .max()?;
                Some((state.shared_event_timeline.event_owned(), newest))
            })
            .collect();
        let queue = self
            .command_queue
            .as_ref()
            .cloned()
            .ok_or("Metal command queue is unavailable")?;
        Ok((queue, None, cross_queue_waits))
    }

    pub(crate) fn set_active_guest_queue(&mut self, queue_id: Option<u64>) {
        self.active_guest_queue = if self.per_queue_guest_batches {
            queue_id
        } else {
            None
        };
        if let Some(queue_id) = self.active_guest_queue
            && let Err(error) = self.ensure_guest_queue_batch_state(queue_id)
        {
            eprintln!("[REMu][METAL] could not initialize guest queue {queue_id}: {error}");
        }
    }

    fn ensure_guest_queue_batch_state(&mut self, queue_id: u64) -> Result<(), String> {
        if self.guest_queue_batches.contains_key(&queue_id) {
            return Ok(());
        }
        let device = self.device.as_ref().ok_or("Metal device is unavailable")?;
        let command_queue = device.new_command_queue();
        command_queue.set_label(&format!("REMu guest command queue {queue_id}"));
        self.guest_queue_batches.insert(
            queue_id,
            GuestQueueBatchState {
                command_queue,
                shared_event_timeline: super::super::SharedEventTimeline::new(
                    device,
                    SharedEventProducerId::new(queue_id),
                ),
                host_retired_fence_value: 0,
                host_dependencies: std::collections::HashMap::new(),
                resource_dependencies: std::collections::HashMap::new(),
                open_batch: None,
                committed_batches: std::collections::VecDeque::new(),
            },
        );
        Ok(())
    }

    pub(crate) fn active_guest_command_queue(&mut self) -> Result<metal::CommandQueue, String> {
        if !self.per_queue_guest_batches {
            return self
                .command_queue
                .as_ref()
                .cloned()
                .ok_or_else(|| "Metal command queue is unavailable".to_owned());
        }
        let queue_id = self
            .active_guest_queue
            .ok_or("guest command has no logical queue identity")?;
        self.ensure_guest_queue_batch_state(queue_id)?;
        Ok(self
            .guest_queue_batches
            .get(&queue_id)
            .expect("queue state was just created")
            .command_queue
            .clone())
    }

    pub(crate) fn ensure_active_open_guest_batch(&mut self) -> Result<(), String> {
        if !self.per_queue_guest_batches {
            let queue = self
                .command_queue
                .as_ref()
                .cloned()
                .ok_or("Metal command queue is unavailable")?;
            self.ensure_open_guest_batch(&queue);
            return Ok(());
        }
        let queue_id = self
            .active_guest_queue
            .ok_or("guest command has no logical queue identity")?;
        self.ensure_guest_queue_batch_state(queue_id)?;
        let state = self
            .guest_queue_batches
            .get_mut(&queue_id)
            .expect("queue state was just created");
        if state.open_batch.is_none() {
            let command_buffer = state.command_queue.new_command_buffer().to_owned();
            command_buffer.set_label(&format!("REMu translated guest batch queue {queue_id}"));
            state.open_batch = Some(new_guest_command_batch(Some(queue_id), command_buffer));
        }
        Ok(())
    }

    pub(crate) fn active_open_guest_batch(&self) -> Option<&GuestCommandBatch> {
        if !self.per_queue_guest_batches {
            return self.open_guest_batch.as_ref();
        }
        self.active_guest_queue
            .and_then(|queue_id| self.guest_queue_batches.get(&queue_id))
            .and_then(|state| state.open_batch.as_ref())
    }

    pub(crate) fn active_open_guest_batch_mut(&mut self) -> Option<&mut GuestCommandBatch> {
        if !self.per_queue_guest_batches {
            return self.open_guest_batch.as_mut();
        }
        self.active_guest_queue
            .and_then(|queue_id| self.guest_queue_batches.get_mut(&queue_id))
            .and_then(|state| state.open_batch.as_mut())
    }

    pub(crate) fn active_open_guest_batch_and_statistics(
        &mut self,
    ) -> (Option<&mut GuestCommandBatch>, &mut PresenterStatistics) {
        let statistics = &mut self.statistics;
        let batch = if !self.per_queue_guest_batches {
            self.open_guest_batch.as_mut()
        } else {
            self.active_guest_queue
                .and_then(|queue_id| self.guest_queue_batches.get_mut(&queue_id))
                .and_then(|state| state.open_batch.as_mut())
        };
        (batch, statistics)
    }

    pub(crate) fn active_committed_guest_batches(
        &self,
    ) -> &std::collections::VecDeque<GuestCommandBatch> {
        if !self.per_queue_guest_batches {
            return &self.committed_guest_batches;
        }
        let queue_id = self
            .active_guest_queue
            .expect("queue-local batch access requires an active guest queue");
        self.guest_queue_batches
            .get(&queue_id)
            .map_or(&self.committed_guest_batches, |state| {
                &state.committed_batches
            })
    }

    pub(crate) fn active_committed_guest_batches_mut(
        &mut self,
    ) -> &mut std::collections::VecDeque<GuestCommandBatch> {
        if !self.per_queue_guest_batches {
            return &mut self.committed_guest_batches;
        }
        let queue_id = self
            .active_guest_queue
            .expect("queue-local batch access requires an active guest queue");
        &mut self
            .guest_queue_batches
            .get_mut(&queue_id)
            .expect("active guest queue has batch state")
            .committed_batches
    }

    pub(crate) fn committed_guest_batch_count(&self) -> usize {
        if !self.per_queue_guest_batches {
            self.committed_guest_batches.len()
        } else {
            self.guest_queue_batches
                .values()
                .map(|state| state.committed_batches.len())
                .sum()
        }
    }

    pub(crate) fn committed_guest_batches_for_diagnostics(
        &self,
    ) -> Vec<(Option<u64>, usize, &GuestCommandBatch)> {
        if !self.per_queue_guest_batches {
            return self
                .committed_guest_batches
                .iter()
                .enumerate()
                .map(|(index, batch)| (None, index, batch))
                .collect();
        }
        let mut queues = self.guest_queue_batches.iter().collect::<Vec<_>>();
        queues.sort_unstable_by_key(|(queue_id, _)| **queue_id);
        queues
            .into_iter()
            .flat_map(|(&queue_id, state)| {
                state
                    .committed_batches
                    .iter()
                    .enumerate()
                    .map(move |(index, batch)| (Some(queue_id), index, batch))
            })
            .collect()
    }

    pub(crate) fn any_guest_batch(
        &self,
        mut predicate: impl FnMut(&GuestCommandBatch) -> bool,
    ) -> bool {
        if !self.per_queue_guest_batches {
            return self.open_guest_batch.as_ref().is_some_and(&mut predicate)
                || self.committed_guest_batches.iter().any(predicate);
        }
        self.guest_queue_batches.values().any(|state| {
            state.open_batch.as_ref().is_some_and(&mut predicate)
                || state.committed_batches.iter().any(&mut predicate)
        })
    }

    pub(crate) fn record_guest_submission_boundary(
        &mut self,
        source: OrderedEventSource,
        sequence: u64,
    ) {
        if !self.per_queue_guest_batches {
            if let Some(batch) = self.open_guest_batch.as_mut() {
                batch.completion_sequences.push(sequence);
            } else if let Some(batch) = self.committed_guest_batches.back_mut() {
                batch.completion_sequences.push(sequence);
            } else {
                self.publish_guest_submission_completion(sequence);
            }
            return;
        }
        let OrderedEventSource::GuestQueue(queue_id) = source else {
            self.publish_guest_submission_completion(sequence);
            return;
        };
        if let Some(state) = self.guest_queue_batches.get_mut(&queue_id) {
            if let Some(batch) = state.open_batch.as_mut() {
                batch.completion_sequences.push(sequence);
                return;
            }
            if let Some(batch) = state.committed_batches.back_mut() {
                batch.completion_sequences.push(sequence);
                return;
            }
        }
        self.publish_guest_submission_completion(sequence);
    }

    pub(crate) fn ensure_open_guest_batch(&mut self, queue: &metal::CommandQueue) {
        if self.open_guest_batch.is_some() {
            return;
        }
        let command_buffer = queue.new_command_buffer().to_owned();
        command_buffer.set_label("REMu translated guest batch");
        self.open_guest_batch = Some(new_guest_command_batch(None, command_buffer));
    }

    pub(crate) fn retire_completed_guest_batches(&mut self) {
        if self.per_queue_guest_batches {
            let mut queue_ids = self.guest_queue_batches.keys().copied().collect::<Vec<_>>();
            queue_ids.sort_unstable();
            for queue_id in queue_ids {
                loop {
                    let terminal = self
                        .guest_queue_batches
                        .get(&queue_id)
                        .and_then(|state| state.committed_batches.front())
                        .is_some_and(|batch| {
                            matches!(
                                batch.command_buffer.status(),
                                MTLCommandBufferStatus::Completed | MTLCommandBufferStatus::Error
                            )
                        });
                    if !terminal {
                        break;
                    }
                    let batch = self
                        .guest_queue_batches
                        .get_mut(&queue_id)
                        .expect("enumerated queue exists")
                        .committed_batches
                        .pop_front()
                        .expect("terminal front exists");
                    let mut timer =
                        TerminalPublicationTimer::start_if(self.terminal_publication_timing);
                    let terminal_status = batch.command_buffer.status();
                    timer.mark(
                        &mut self.statistics.terminal_publication,
                        TerminalPublicationPhase::MetalTerminal,
                    );
                    let execution_error =
                        (terminal_status == MTLCommandBufferStatus::Error).then(|| {
                            command_buffer_failure_message(
                                &format!(
                                    "Metal guest command batch queue={} [{}]",
                                    queue_id,
                                    describe_guest_batch(&batch.work)
                                ),
                                &batch.command_buffer,
                            )
                        });
                    timer.mark(
                        &mut self.statistics.terminal_publication,
                        TerminalPublicationPhase::TerminalMetadata,
                    );
                    let result = self.complete_terminal_guest_batch(&batch, execution_error, timer);
                    if let Err(error) = result {
                        eprintln!(
                            "[REMu][METAL] queue {queue_id} guest batch finalization failed: {error}"
                        );
                    }
                }
            }
            return;
        }
        loop {
            let Some(batch) = self.committed_guest_batches.front() else {
                return;
            };
            match batch.command_buffer.status() {
                MTLCommandBufferStatus::Completed => {
                    let batch = self
                        .committed_guest_batches
                        .pop_front()
                        .expect("front batch exists");
                    let mut timer =
                        TerminalPublicationTimer::start_if(self.terminal_publication_timing);
                    let terminal_status = (cfg!(debug_assertions)
                        || self.terminal_publication_timing)
                        .then(|| batch.command_buffer.status());
                    timer.mark(
                        &mut self.statistics.terminal_publication,
                        TerminalPublicationPhase::MetalTerminal,
                    );
                    debug_assert_eq!(terminal_status, Some(MTLCommandBufferStatus::Completed));
                    timer.mark(
                        &mut self.statistics.terminal_publication,
                        TerminalPublicationPhase::TerminalMetadata,
                    );
                    let result = self.complete_terminal_guest_batch(&batch, None, timer);
                    if let Err(error) = result {
                        eprintln!(
                            "[REMu][METAL] completed guest batch finalization failed: {error}"
                        );
                    }
                }
                MTLCommandBufferStatus::Error => {
                    let batch = self
                        .committed_guest_batches
                        .pop_front()
                        .expect("front batch exists");
                    let mut timer =
                        TerminalPublicationTimer::start_if(self.terminal_publication_timing);
                    let terminal_status = (cfg!(debug_assertions)
                        || self.terminal_publication_timing)
                        .then(|| batch.command_buffer.status());
                    timer.mark(
                        &mut self.statistics.terminal_publication,
                        TerminalPublicationPhase::MetalTerminal,
                    );
                    debug_assert_eq!(terminal_status, Some(MTLCommandBufferStatus::Error));
                    let execution_error = command_buffer_failure_message(
                        &format!(
                            "Metal guest command batch [{}]",
                            describe_guest_batch(&batch.work)
                        ),
                        &batch.command_buffer,
                    );
                    timer.mark(
                        &mut self.statistics.terminal_publication,
                        TerminalPublicationPhase::TerminalMetadata,
                    );
                    let result =
                        self.complete_terminal_guest_batch(&batch, Some(execution_error), timer);
                    // A terminal native failure cannot be retried in-place. Publish both its
                    // ordered CPU labels and submission ID after reporting the error so the guest
                    // observes one failed queue item instead of a producerless 120-second wait.
                    if let Err(error) = result {
                        eprintln!("[REMu][METAL] {error}");
                    }
                }
                _ => return,
            }
        }
    }

    pub(crate) fn finish_terminal_guest_batch(
        &mut self,
        batch: &GuestCommandBatch,
        execution_error: Option<String>,
        timer: &mut TerminalPublicationTimer,
    ) -> Result<(), String> {
        let execution_succeeded = execution_error.is_none();
        let mut result = execution_error.map_or(Ok(()), Err);
        let error_fence_advance = (|| {
            if !execution_succeeded
                && let (Some(queue_id), Some(maximum_fence)) =
                    (batch.queue_id, batch.max_encoded_signal)
            {
                let producer = SharedEventProducerId::new(queue_id);
                let state = self
                    .guest_queue_batches
                    .get_mut(&queue_id)
                    .ok_or_else(|| format!("missing terminal guest queue {queue_id}"))?;
                if let Some(value) = terminal_shared_event_advance(
                    SharedEventBatchTerminalState::Error,
                    producer,
                    state.shared_event_timeline.event().signaled_value(),
                    Some(maximum_fence),
                )
                .map_err(|error| error.to_string())?
                {
                    state
                        .shared_event_timeline
                        .event()
                        .set_signaled_value(value);
                    self.statistics.shared_event_error_advances = self
                        .statistics
                        .shared_event_error_advances
                        .saturating_add(1);
                }
            }
            Ok::<(), String>(())
        })();
        timer.mark(
            &mut self.statistics.terminal_publication,
            TerminalPublicationPhase::ErrorFenceAdvance,
        );
        error_fence_advance?;
        if execution_succeeded {
            let writeback = write_back_pending_globals(&batch.global_writebacks);
            if let Ok(applied) = &writeback {
                self.record_produced_global_writebacks(applied);
            }
            result = merge_guest_batch_result(result, "global writeback", writeback.map(|_| ()));
        }
        timer.mark(
            &mut self.statistics.terminal_publication,
            TerminalPublicationPhase::GlobalWriteback,
        );
        // RELEASE_MEM, WRITE_DATA, and DMA fills are command-processor effects ordered after the
        // native work, not shader results. Once the Metal command buffer reaches either terminal
        // state, publish them even when the draw/dispatch or an unrelated global readback failed.
        // This matches the existing policy of retiring the submission after a native error while
        // retaining the error as the primary result.
        let result = self.publish_terminal_guest_memory_writes(&batch.memory_writes, result);
        timer.mark(
            &mut self.statistics.terminal_publication,
            TerminalPublicationPhase::GuestMemoryWrites,
        );
        if let (Some(queue_id), Some(maximum_fence)) = (batch.queue_id, batch.max_encoded_signal) {
            let state = self
                .guest_queue_batches
                .get_mut(&queue_id)
                .expect("terminal queue state remains live");
            state.host_retired_fence_value =
                state.host_retired_fence_value.max(maximum_fence.value());
        }
        timer.mark(
            &mut self.statistics.terminal_publication,
            TerminalPublicationPhase::QueueRetirement,
        );
        result
    }

    fn complete_terminal_guest_batch(
        &mut self,
        batch: &GuestCommandBatch,
        execution_error: Option<String>,
        mut timer: TerminalPublicationTimer,
    ) -> Result<(), String> {
        self.record_shared_event_ack_terminal(&batch.command_buffer);
        let result = self.finish_terminal_guest_batch(batch, execution_error, &mut timer);
        self.statistics.gpu_execution_time +=
            completed_command_buffer_gpu_duration(&batch.command_buffer);
        timer.mark(
            &mut self.statistics.terminal_publication,
            TerminalPublicationPhase::GpuTimeQuery,
        );
        self.publish_guest_batch_completion(batch);
        timer.mark(
            &mut self.statistics.terminal_publication,
            TerminalPublicationPhase::CompletionWake,
        );
        result
    }

    pub(super) fn gpu_events_per_tick_budget(&self) -> usize {
        if remu_gpu::gpu_work_items_pending() > super::super::GPU_EVENTS_PER_TICK_BACKLOG_THRESHOLD
        {
            super::super::GPU_EVENTS_PER_TICK
                .saturating_mul(super::super::GPU_EVENTS_PER_TICK_BACKLOG_MULTIPLIER)
        } else {
            super::super::GPU_EVENTS_PER_TICK
        }
    }

    pub(super) fn guest_submission_backpressure_full(&self) -> bool {
        self.committed_guest_batch_count() >= super::super::MAX_COMMITTED_GUEST_BATCHES
            && self
                .active_open_guest_batch()
                .is_none_or(|batch| batch.work.len() >= super::super::GUEST_BATCH_WORK_CAPACITY)
    }

    pub(crate) fn flush_guest_command_batch(
        &mut self,
        policy: FlushPolicy,
        reason: GuestBatchFlushReason,
    ) -> Result<(), String> {
        if self.per_queue_guest_batches {
            let deadline = Instant::now() + COMMAND_BUFFER_SETTLE_TIMEOUT;
            if reason == GuestBatchFlushReason::EventLoop {
                return self.flush_all_guest_command_batches_by(policy, reason, deadline);
            }
            let queue_id = self
                .active_guest_queue
                .ok_or("queue-local guest batch flush has no active guest queue")?;
            return self.flush_guest_queue_command_batch_by(queue_id, policy, reason, deadline);
        }
        self.flush_guest_command_batch_by(
            policy,
            reason,
            Instant::now() + COMMAND_BUFFER_SETTLE_TIMEOUT,
        )
    }

    /// Flushes the open batch, bounding any budget-reclaiming settle by `settle_deadline`.
    fn flush_guest_command_batch_by(
        &mut self,
        policy: FlushPolicy,
        reason: GuestBatchFlushReason,
        settle_deadline: Instant,
    ) -> Result<(), String> {
        if self.open_guest_batch.is_none() {
            return Ok(());
        }
        // A batch that outlives the settle budget stays queued, so the in-flight count can still be
        // at the cap below. Committing one batch past it is bounded and temporary; declining to
        // commit would hand a `Required` caller the uncommitted results it is about to read.
        let mut settlement = Ok(());
        if self.committed_guest_batches.len() >= super::super::MAX_COMMITTED_GUEST_BATCHES {
            match policy {
                FlushPolicy::Opportunistic => return Ok(()),
                FlushPolicy::Required => {
                    settlement = match self.settle_front_guest_batch(settle_deadline) {
                        FrontBatchSettlement::Empty => Ok(()),
                        FrontBatchSettlement::Retired(result) => result,
                        FrontBatchSettlement::Pending(error) => Err(error),
                    };
                }
            }
        }
        let Some(mut batch) = self.open_guest_batch.take() else {
            return Ok(());
        };
        match reason {
            GuestBatchFlushReason::EventLoop => {
                self.statistics.batch_flushes_from_event_loop = self
                    .statistics
                    .batch_flushes_from_event_loop
                    .saturating_add(1);
            }
            GuestBatchFlushReason::BlockedWait => {
                self.statistics.batch_flushes_for_blocked_wait = self
                    .statistics
                    .batch_flushes_for_blocked_wait
                    .saturating_add(1);
            }
            GuestBatchFlushReason::SharedEventHostEffect => {
                self.statistics.batch_flushes_for_shared_event_host_effect = self
                    .statistics
                    .batch_flushes_for_shared_event_host_effect
                    .saturating_add(1);
            }
            GuestBatchFlushReason::FlipSnapshot => {
                self.statistics.batch_flushes_for_flip_snapshot = self
                    .statistics
                    .batch_flushes_for_flip_snapshot
                    .saturating_add(1);
            }
            GuestBatchFlushReason::WorkCapacity => {
                self.statistics.batch_flushes_for_work_capacity = self
                    .statistics
                    .batch_flushes_for_work_capacity
                    .saturating_add(1);
            }
            GuestBatchFlushReason::CommandProcessorBarrier => {
                self.statistics.batch_flushes_for_command_processor_barrier = self
                    .statistics
                    .batch_flushes_for_command_processor_barrier
                    .saturating_add(1);
            }
            GuestBatchFlushReason::CpuSynchronization => {
                self.statistics.batch_flushes_for_cpu_synchronization = self
                    .statistics
                    .batch_flushes_for_cpu_synchronization
                    .saturating_add(1);
            }
        }
        batch.end_compute_pass();
        if batch.end_render_pass() {
            self.statistics.render_pass_ends_for_batch_flush = self
                .statistics
                .render_pass_ends_for_batch_flush
                .saturating_add(1);
        }
        batch.committed_at = Some(Instant::now());
        if self.per_queue_guest_batches
            && let Some(waker) = self.event_loop_waker.clone()
        {
            attach_guest_batch_completion_waker(&batch.command_buffer, waker);
        }
        batch.command_buffer.commit();
        self.statistics.submitted_guest_batches =
            self.statistics.submitted_guest_batches.saturating_add(1);
        self.committed_guest_batches.push_back(batch);
        settlement
    }

    fn flush_guest_queue_command_batch_by(
        &mut self,
        queue_id: u64,
        policy: FlushPolicy,
        reason: GuestBatchFlushReason,
        settle_deadline: Instant,
    ) -> Result<(), String> {
        if self
            .guest_queue_batches
            .get(&queue_id)
            .is_none_or(|state| state.open_batch.is_none())
        {
            return Ok(());
        }
        self.retire_completed_guest_batches();
        let mut settlement = Ok(());
        let committed_before_reclaim = self.committed_guest_batch_count();
        if committed_before_reclaim >= super::super::MAX_COMMITTED_GUEST_BATCHES {
            match policy {
                FlushPolicy::Opportunistic => return Ok(()),
                FlushPolicy::Required => {
                    let settled = self.settle_next_global_guest_batch(settle_deadline);
                    if committed_before_reclaim > super::super::MAX_COMMITTED_GUEST_BATCHES
                        && let FrontBatchSettlement::Pending(error) = &settled
                    {
                        // A required flush may temporarily occupy the legacy one emergency slot,
                        // but that slot is global. Do not grow one extra batch per guest queue when
                        // the device has already exhausted the shared settle deadline.
                        return Err(error.clone());
                    }
                    settlement = match settled {
                        FrontBatchSettlement::Empty => Ok(()),
                        FrontBatchSettlement::Retired(result) => result,
                        FrontBatchSettlement::Pending(error) => Err(error),
                    };
                }
            }
        }
        let completion_waker = self.event_loop_waker.clone();
        let state = self
            .guest_queue_batches
            .get_mut(&queue_id)
            .expect("open queue state exists");
        let Some(mut batch) = state.open_batch.take() else {
            return settlement;
        };
        let mut acknowledgement_cpu = self
            .shared_event_ack_batch_cpu
            .remove(&queue_id)
            .unwrap_or_default();
        let acknowledgement_trace = self.pending_shared_event_ack_traces.remove(&queue_id);
        if let Some(trace) = &acknowledgement_trace {
            acknowledgement_cpu.translation = acknowledgement_cpu
                .translation
                .saturating_add(trace.host_preflight);
        }
        batch.end_compute_pass();
        if batch.end_render_pass() {
            self.statistics.render_pass_ends_for_batch_flush = self
                .statistics
                .render_pass_ends_for_batch_flush
                .saturating_add(1);
        }
        let committed_at = Instant::now();
        batch.committed_at = Some(committed_at);
        if let Some(waker) = completion_waker {
            attach_guest_batch_completion_waker(&batch.command_buffer, waker);
        }
        let native_host = acknowledgement_trace
            .as_ref()
            .map(|_| Arc::new(Mutex::new(SharedEventAckNativeHostTimes::default())));
        if let Some(native_host) = native_host.as_ref() {
            attach_shared_event_ack_phase_handlers(&batch.command_buffer, Arc::clone(native_host));
        }
        let commit_started = Instant::now();
        batch.command_buffer.commit();
        let commit_call = Instant::now().saturating_duration_since(commit_started);
        if let (Some(trace), Some(native_host)) = (acknowledgement_trace, native_host) {
            let command_buffer_key = batch.command_buffer.as_ptr() as usize;
            self.shared_event_ack_trace_buffers_by_fence
                .insert(trace.fence, command_buffer_key);
            self.shared_event_ack_traces.insert(
                command_buffer_key,
                SharedEventAckTrace {
                    fence: trace.fence,
                    committed_at,
                    event_queue: trace.event_queue,
                    cpu: acknowledgement_cpu,
                    ack_encoding: trace.ack_encoding,
                    commit_call,
                    native_host,
                    terminal: None,
                    ack_observed_at: None,
                },
            );
        }
        state.committed_batches.push_back(batch);
        self.record_guest_batch_flush(reason);
        settlement
    }

    fn flush_all_guest_command_batches_by(
        &mut self,
        policy: FlushPolicy,
        reason: GuestBatchFlushReason,
        deadline: Instant,
    ) -> Result<(), String> {
        let mut queue_ids = self.guest_queue_batches.keys().copied().collect::<Vec<_>>();
        queue_ids.sort_unstable();
        let mut result = Ok(());
        for queue_id in queue_ids {
            let flushed =
                self.flush_guest_queue_command_batch_by(queue_id, policy, reason, deadline);
            result = merge_guest_batch_result(result, "later guest queue flush", flushed);
        }
        result
    }

    fn record_guest_batch_flush(&mut self, reason: GuestBatchFlushReason) {
        match reason {
            GuestBatchFlushReason::EventLoop => {
                self.statistics.batch_flushes_from_event_loop = self
                    .statistics
                    .batch_flushes_from_event_loop
                    .saturating_add(1);
            }
            GuestBatchFlushReason::BlockedWait => {
                self.statistics.batch_flushes_for_blocked_wait = self
                    .statistics
                    .batch_flushes_for_blocked_wait
                    .saturating_add(1);
            }
            GuestBatchFlushReason::SharedEventHostEffect => {
                self.statistics.batch_flushes_for_shared_event_host_effect = self
                    .statistics
                    .batch_flushes_for_shared_event_host_effect
                    .saturating_add(1);
            }
            GuestBatchFlushReason::FlipSnapshot => {
                self.statistics.batch_flushes_for_flip_snapshot = self
                    .statistics
                    .batch_flushes_for_flip_snapshot
                    .saturating_add(1);
            }
            GuestBatchFlushReason::WorkCapacity => {
                self.statistics.batch_flushes_for_work_capacity = self
                    .statistics
                    .batch_flushes_for_work_capacity
                    .saturating_add(1);
            }
            GuestBatchFlushReason::CommandProcessorBarrier => {
                self.statistics.batch_flushes_for_command_processor_barrier = self
                    .statistics
                    .batch_flushes_for_command_processor_barrier
                    .saturating_add(1);
            }
            GuestBatchFlushReason::CpuSynchronization => {
                self.statistics.batch_flushes_for_cpu_synchronization = self
                    .statistics
                    .batch_flushes_for_cpu_synchronization
                    .saturating_add(1);
            }
        }
        self.statistics.submitted_guest_batches =
            self.statistics.submitted_guest_batches.saturating_add(1);
    }

    /// Settles the oldest committed batch and publishes its completion effects.
    ///
    /// A batch that outlives `deadline` is put back at the front: later buffers from the same queue
    /// cannot complete before it, so none of their CPU-side completion effects may overtake it.
    fn settle_front_guest_batch(&mut self, deadline: Instant) -> FrontBatchSettlement {
        let Some(batch) = self.committed_guest_batches.pop_front() else {
            return FrontBatchSettlement::Empty;
        };
        let mut timer = TerminalPublicationTimer::start_if(self.terminal_publication_timing);
        let execution_error = match settle_guest_batch_by(&batch, deadline) {
            GuestBatchSettlement::Completed => None,
            GuestBatchSettlement::Failed(error) => Some(error),
            GuestBatchSettlement::Pending(error) => {
                self.committed_guest_batches.push_front(batch);
                return FrontBatchSettlement::Pending(error);
            }
        };
        timer.mark(
            &mut self.statistics.terminal_publication,
            TerminalPublicationPhase::MetalTerminal,
        );
        timer.mark(
            &mut self.statistics.terminal_publication,
            TerminalPublicationPhase::TerminalMetadata,
        );
        let result = self.complete_terminal_guest_batch(&batch, execution_error, timer);
        FrontBatchSettlement::Retired(result)
    }

    fn settle_front_guest_queue_batch(
        &mut self,
        queue_id: u64,
        deadline: Instant,
    ) -> FrontBatchSettlement {
        let Some(batch) = self
            .guest_queue_batches
            .get_mut(&queue_id)
            .and_then(|state| state.committed_batches.pop_front())
        else {
            return FrontBatchSettlement::Empty;
        };
        let mut timer = TerminalPublicationTimer::start_if(self.terminal_publication_timing);
        let execution_error = match settle_guest_batch_by(&batch, deadline) {
            GuestBatchSettlement::Completed => None,
            GuestBatchSettlement::Failed(error) => Some(error),
            GuestBatchSettlement::Pending(error) => {
                self.guest_queue_batches
                    .get_mut(&queue_id)
                    .expect("settled queue exists")
                    .committed_batches
                    .push_front(batch);
                return FrontBatchSettlement::Pending(error);
            }
        };
        timer.mark(
            &mut self.statistics.terminal_publication,
            TerminalPublicationPhase::MetalTerminal,
        );
        timer.mark(
            &mut self.statistics.terminal_publication,
            TerminalPublicationPhase::TerminalMetadata,
        );
        let result = self.complete_terminal_guest_batch(&batch, execution_error, timer);
        FrontBatchSettlement::Retired(result)
    }

    fn next_global_guest_batch_queue(&self) -> Option<u64> {
        self.guest_queue_batches
            .iter()
            .filter_map(|(&queue_id, state)| {
                state
                    .committed_batches
                    .front()
                    .map(|batch| (queue_id, batch))
            })
            .min_by_key(|(queue_id, batch)| {
                (
                    !matches!(
                        batch.command_buffer.status(),
                        MTLCommandBufferStatus::Completed | MTLCommandBufferStatus::Error
                    ),
                    batch.committed_at,
                    *queue_id,
                )
            })
            .map(|(queue_id, _)| queue_id)
    }

    fn settle_next_global_guest_batch(&mut self, deadline: Instant) -> FrontBatchSettlement {
        self.next_global_guest_batch_queue()
            .map_or(FrontBatchSettlement::Empty, |queue_id| {
                self.settle_front_guest_queue_batch(queue_id, deadline)
            })
    }

    pub(crate) fn wait_for_guest_command_batches(&mut self) -> Result<(), String> {
        self.wait_for_guest_command_batches_within(COMMAND_BUFFER_SETTLE_TIMEOUT)
    }

    /// Commits and settles only the logical guest queue currently being translated.
    ///
    /// Synchronous diagnostics need the render pass containing the sampled draw to be complete,
    /// but draining unrelated queues can deadlock behind guest-side cross-queue dependencies. The
    /// normal CPU synchronization path intentionally drains every queue; this narrower path is for
    /// observations whose resources are owned by the active queue itself.
    pub(crate) fn wait_for_active_guest_queue_command_batches(&mut self) -> Result<(), String> {
        if !self.per_queue_guest_batches {
            return self.wait_for_guest_command_batches();
        }
        self.guest_batch_cpu_synchronization_epoch =
            self.guest_batch_cpu_synchronization_epoch.saturating_add(1);
        let queue_id = self
            .active_guest_queue
            .ok_or("active-queue synchronization has no logical guest queue")?;
        let deadline = Instant::now() + COMMAND_BUFFER_SETTLE_TIMEOUT;
        let mut result = self.flush_guest_queue_command_batch_by(
            queue_id,
            FlushPolicy::Required,
            GuestBatchFlushReason::CpuSynchronization,
            deadline,
        );
        debug_assert!(
            result.is_err()
                || self
                    .guest_queue_batches
                    .get(&queue_id)
                    .is_none_or(|state| state.open_batch.is_none())
        );
        loop {
            match self.settle_front_guest_queue_batch(queue_id, deadline) {
                FrontBatchSettlement::Empty => return result,
                FrontBatchSettlement::Retired(batch_result) => {
                    result = merge_guest_batch_result(
                        result,
                        "later active guest queue batch",
                        batch_result,
                    );
                }
                FrontBatchSettlement::Pending(error) => {
                    return merge_guest_batch_result(
                        result,
                        "pending active guest queue batch",
                        Err(error),
                    );
                }
            }
        }
    }

    pub(crate) fn wait_for_guest_command_batches_within(
        &mut self,
        timeout: Duration,
    ) -> Result<(), String> {
        self.guest_batch_cpu_synchronization_epoch =
            self.guest_batch_cpu_synchronization_epoch.saturating_add(1);
        if self.per_queue_guest_batches {
            return self.wait_for_guest_queue_command_batches_within(timeout);
        }
        // One budget covers the whole drain, including the flush's own slot reclaim. Charging every
        // queued batch its own timeout would sum to far past the host watchdog window this bound
        // exists to stay inside, and a device that stalled on the first batch will not retire the
        // rest either.
        let deadline = Instant::now() + timeout;
        self.flush_guest_command_batch_by(
            FlushPolicy::Required,
            GuestBatchFlushReason::CpuSynchronization,
            deadline,
        )?;
        debug_assert!(self.open_guest_batch.is_none());
        let mut result = Ok(());
        loop {
            match self.settle_front_guest_batch(deadline) {
                FrontBatchSettlement::Empty => return result,
                // Drain every already-committed batch even after one fails. Returning early here
                // used to abandon later RELEASE_MEM labels and completion IDs during shutdown or a
                // synchronous readback, converting a preserved Metal error into an unrelated guest
                // RenderThread timeout.
                FrontBatchSettlement::Retired(batch_result) => {
                    result =
                        merge_guest_batch_result(result, "later ordered guest batch", batch_result);
                }
                FrontBatchSettlement::Pending(error) => {
                    return merge_guest_batch_result(
                        result,
                        "pending ordered guest batch",
                        Err(error),
                    );
                }
            }
        }
    }

    fn wait_for_guest_queue_command_batches_within(
        &mut self,
        timeout: Duration,
    ) -> Result<(), String> {
        let deadline = Instant::now() + timeout;
        let mut result = self.flush_all_guest_command_batches_by(
            FlushPolicy::Required,
            GuestBatchFlushReason::CpuSynchronization,
            deadline,
        );
        debug_assert!(
            result.is_err()
                || self
                    .guest_queue_batches
                    .values()
                    .all(|state| state.open_batch.is_none())
        );
        loop {
            let mut queue_ids = self
                .guest_queue_batches
                .iter()
                .filter_map(|(&queue_id, state)| {
                    (!state.committed_batches.is_empty()).then_some(queue_id)
                })
                .collect::<Vec<_>>();
            queue_ids.sort_unstable();
            if queue_ids.is_empty() {
                return result;
            }
            let mut retired_any = false;
            let mut pending_error = None;
            for queue_id in queue_ids {
                match self.settle_front_guest_queue_batch(queue_id, deadline) {
                    FrontBatchSettlement::Empty => {}
                    FrontBatchSettlement::Retired(batch_result) => {
                        retired_any = true;
                        result = merge_guest_batch_result(
                            result,
                            "later ordered guest queue batch",
                            batch_result,
                        );
                    }
                    FrontBatchSettlement::Pending(error) => {
                        pending_error.get_or_insert(error);
                    }
                }
            }
            if !retired_any {
                return match pending_error {
                    Some(error) => merge_guest_batch_result(
                        result,
                        "pending ordered guest queue batch",
                        Err(error),
                    ),
                    None => result,
                };
            }
        }
    }
}

fn attach_guest_batch_completion_waker(
    command_buffer: &CommandBuffer,
    waker: winit::event_loop::EventLoopProxy<()>,
) {
    let handler = ConcreteBlock::new(move |_| {
        let _ = waker.send_event(());
    })
    .copy();
    command_buffer.add_completed_handler(&handler);
}

fn attach_shared_event_ack_phase_handlers(
    command_buffer: &CommandBuffer,
    native_host: Arc<Mutex<SharedEventAckNativeHostTimes>>,
) {
    let scheduled_times: Arc<Mutex<SharedEventAckNativeHostTimes>> = Arc::clone(&native_host);
    let scheduled = ConcreteBlock::new(move |_| {
        scheduled_times
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .scheduled_at = Some(Instant::now());
    })
    .copy();
    command_buffer.add_scheduled_handler(&scheduled);

    let completed = ConcreteBlock::new(move |_| {
        native_host
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .completed_at = Some(Instant::now());
    })
    .copy();
    command_buffer.add_completed_handler(&completed);
}

fn new_guest_command_batch(
    queue_id: Option<u64>,
    command_buffer: CommandBuffer,
) -> GuestCommandBatch {
    GuestCommandBatch {
        queue_id,
        command_buffer,
        work: Vec::new(),
        committed_at: None,
        global_upload_chunks: Vec::new(),
        global_buffer_versions: Vec::new(),
        global_writebacks: Vec::new(),
        guest_write_ranges: Vec::new(),
        memory_writes: Vec::new(),
        resident_write_footprint: Vec::new(),
        resident_texture_versions: Vec::new(),
        resident_write_footprint_complete: true,
        render_pass: None,
        compute_encoder: None,
        max_encoded_signal: None,
        signal_checkpoints: Vec::new(),
        completion_sequences: Vec::new(),
    }
}

/// Waits for one committed guest batch until `deadline` and describes why it failed, when it did.
///
/// A batch that outlives the budget is worded differently from a driver-reported failure: its work
/// is still outstanding, so callers must retain it rather than treat it as having run.
fn settle_guest_batch_by(batch: &GuestCommandBatch, deadline: Instant) -> GuestBatchSettlement {
    let budget = deadline.saturating_duration_since(Instant::now());
    let settled = wait_until_settled_within(&batch.command_buffer, budget);
    // Only the timeout and failure branches need the batch described, and both are rare. The
    // common path — a batch that completed within budget — must not pay for formatting every
    // draw in the stream, so the description is built lazily after the outcome is known.
    let describe = || {
        let queue = batch
            .queue_id
            .map_or_else(String::new, |queue_id| format!(" queue={queue_id}"));
        format!(
            "Metal guest command batch{queue} [{}]",
            describe_guest_batch(&batch.work)
        )
    };
    if !settled {
        return GuestBatchSettlement::Pending(settle_timeout_message(
            &describe(),
            &batch.command_buffer,
        ));
    }
    if batch.command_buffer.status() == MTLCommandBufferStatus::Completed {
        GuestBatchSettlement::Completed
    } else {
        GuestBatchSettlement::Failed(command_buffer_failure_message(
            &describe(),
            &batch.command_buffer,
        ))
    }
}

pub(crate) fn describe_guest_batch(work: &[GuestBatchWork]) -> String {
    work.iter()
        .map(|item| match item {
            GuestBatchWork::Draw {
                sequence,
                indexed,
                indirect_arguments_address,
                export,
                pixel,
                index_count,
                instance_count,
                target,
                extent,
                export_msl_bytes,
                pixel_msl_bytes,
                export_dispatcher,
                pixel_dispatcher,
            } => {
                let source = indirect_arguments_address.map_or_else(
                    || "direct".to_owned(),
                    |address| format!("indirect@{address:#x}"),
                );
                format!(
                    "draw(seq={sequence},indexed={indexed},source={source},es={export:#x}/{export_msl_bytes}B/dispatcher={export_dispatcher},ps={pixel:#x}/{pixel_msl_bytes}B/dispatcher={pixel_dispatcher},indices={index_count},instances={instance_count},target={target:#x}/{extent:?})"
                )
            }
            GuestBatchWork::Compute {
                sequence,
                shader,
                source,
                grid,
                threads_per_group,
                msl_bytes,
                dispatcher,
            } => format!(
                "compute(seq={sequence},cs={shader:#x}/{msl_bytes}B/dispatcher={dispatcher},source={source:?},grid={grid:?},local={threads_per_group:?})"
            ),
        })
        .collect::<Vec<_>>()
        .join(",")
}

pub(crate) fn merge_guest_batch_result(
    current: Result<(), String>,
    operation: &str,
    next: Result<(), String>,
) -> Result<(), String> {
    match (current, next) {
        (Ok(()), Ok(())) => Ok(()),
        (Err(error), Ok(())) => Err(error),
        (Ok(()), Err(error)) => Err(format!("{operation}: {error}")),
        (Err(primary), Err(secondary)) => {
            Err(format!("{primary}; {operation} also failed: {secondary}"))
        }
    }
}

impl GuestCommandBatch {
    fn reserve_global_upload(
        &mut self,
        device: &Device,
        byte_count: usize,
    ) -> Result<(Buffer, usize), String> {
        let byte_count = byte_count.max(1);
        let aligned_byte_count = byte_count
            .checked_add(GLOBAL_UPLOAD_ALIGNMENT - 1)
            .map(|value| value & !(GLOBAL_UPLOAD_ALIGNMENT - 1))
            .ok_or("global upload size overflows")?;
        let needs_chunk = self.global_upload_chunks.last().is_none_or(|chunk| {
            chunk
                .used
                .checked_add(aligned_byte_count)
                .is_none_or(|end| end > chunk.buffer.length() as usize)
        });
        if needs_chunk {
            let chunk_byte_count = GLOBAL_UPLOAD_CHUNK_BYTE_COUNT.max(aligned_byte_count);
            let length = u64::try_from(chunk_byte_count)
                .map_err(|_| "global upload chunk size does not fit Metal")?;
            let buffer = device.new_buffer(length, MTLResourceOptions::StorageModeShared);
            buffer.set_label("REMu guest-batch global upload arena");
            self.global_upload_chunks
                .push(GlobalUploadChunk { buffer, used: 0 });
        }
        let chunk = self
            .global_upload_chunks
            .last_mut()
            .expect("global upload chunk was created");
        let offset = chunk.used;
        chunk.used = offset + aligned_byte_count;
        Ok((chunk.buffer.clone(), offset))
    }

    pub(crate) fn upload_global_bytes(
        &mut self,
        device: &Device,
        bytes: &[u8],
    ) -> Result<(Buffer, u64), String> {
        let (buffer, offset) = self.reserve_global_upload(device, bytes.len())?;
        if !bytes.is_empty() && !super::super::MetalBytes::new(&buffer).write(offset, bytes) {
            return Err(format!(
                "global upload range {offset}+{} exceeds Metal buffer length {}",
                bytes.len(),
                buffer.length()
            ));
        }
        Ok((
            buffer,
            u64::try_from(offset).map_err(|_| "global upload offset does not fit Metal")?,
        ))
    }

    /// Reserves batch-owned shared storage that a later Metal encoder initializes in full.
    pub(crate) fn allocate_global_bytes(
        &mut self,
        device: &Device,
        byte_count: usize,
    ) -> Result<(Buffer, u64), String> {
        let (buffer, offset) = self.reserve_global_upload(device, byte_count)?;
        Ok((
            buffer,
            u64::try_from(offset).map_err(|_| "global allocation offset does not fit Metal")?,
        ))
    }

    /// Copies one live guest-physical range directly into this batch's shared upload arena.
    ///
    /// The fallback global-buffer path previously copied the same bytes twice: backing mmap to a
    /// temporary `Vec`, then that `Vec` into Metal. Reading into the final arena keeps the same
    /// validated backing semantics while removing the second full-range copy and allocation.
    pub(crate) fn upload_guest_global_bytes(
        &mut self,
        device: &Device,
        guest_address: u64,
        byte_count: usize,
    ) -> Result<(Buffer, u64), String> {
        if byte_count == 0 {
            return Err("guest global upload is empty".to_owned());
        }
        let (buffer, offset) = self.reserve_global_upload(device, byte_count)?;
        // `reserve_global_upload` retained a shared Metal buffer with at least `byte_count` bytes
        // at `offset`. This mutable window is used only for the synchronous backing copy below.
        let bytes = super::super::MetalBytes::new(&buffer);
        let destination = bytes.slice_mut(offset, byte_count).ok_or_else(|| {
            format!(
                "guest global upload range {offset}+{byte_count} exceeds Metal buffer length {}",
                buffer.length()
            )
        })?;
        if !read_guest_shared_backing(guest_address, destination) {
            return Err(format!(
                "synchronized global binding {guest_address:#x}/{byte_count} has no readable guest backing"
            ));
        }
        Ok((
            buffer,
            u64::try_from(offset).map_err(|_| "global upload offset does not fit Metal")?,
        ))
    }

    pub(crate) fn end_render_pass(&mut self) -> bool {
        if let Some(pass) = self.render_pass.take() {
            pass.encoder.end_encoding();
            true
        } else {
            false
        }
    }

    pub(crate) fn close_active_pass_encoders(&mut self) {
        self.end_compute_pass();
        self.end_render_pass();
    }

    pub(crate) fn end_compute_pass(&mut self) -> bool {
        if let Some(encoder) = self.compute_encoder.take() {
            encoder.end_encoding();
            true
        } else {
            false
        }
    }

    pub(crate) fn ensure_compute_encoder(&mut self) -> &ComputeCommandEncoder {
        if self.compute_encoder.is_none() {
            self.compute_encoder =
                Some(self.command_buffer.new_compute_command_encoder().to_owned());
        }
        self.compute_encoder
            .as_ref()
            .expect("compute encoder was just opened")
    }
}

pub(crate) fn guest_batch_writes_overlap(
    batch: &GuestCommandBatch,
    address: u64,
    byte_count: u64,
) -> bool {
    batch.global_buffer_versions.iter().any(|version| {
        guest_physical_ranges_overlap(
            version.guest_address,
            version.byte_count,
            address,
            byte_count,
        )
    }) || batch
        .guest_write_ranges
        .iter()
        .any(|range| pending_guest_write_range_overlaps(range, address, byte_count))
        || batch
            .global_writebacks
            .iter()
            .any(|writeback| pending_global_writeback_overlaps(writeback, address, byte_count))
        || batch.memory_writes.iter().any(|write| {
            let (write_address, write_bytes) = write.range();
            guest_physical_ranges_overlap(write_address, write_bytes, address, byte_count)
        })
}

fn pending_global_writeback_overlaps(
    writeback: &PendingGlobalWriteback,
    address: u64,
    byte_count: u64,
) -> bool {
    let Ok(writeback_byte_count) = u64::try_from(writeback.byte_count) else {
        return true;
    };
    if guest_ranges_overlap(
        writeback.guest_address,
        writeback_byte_count,
        address,
        byte_count,
    ) {
        return true;
    }
    let Some(producer_identity) = writeback.mapping_identity() else {
        return guest_physical_ranges_overlap(
            writeback.guest_address,
            writeback_byte_count,
            address,
            byte_count,
        );
    };
    let Some(read_identity) = guest_shared_backing_identity(address, byte_count) else {
        return false;
    };
    producer_identity.backing_id == read_identity.backing_id
        && guest_ranges_overlap(
            producer_identity.backing_offset,
            writeback_byte_count,
            read_identity.backing_offset,
            byte_count,
        )
}

fn pending_guest_write_range_overlaps(
    range: &PendingGuestWriteRange,
    address: u64,
    byte_count: u64,
) -> bool {
    if guest_ranges_overlap(range.address, range.byte_count, address, byte_count) {
        return true;
    }
    let Some(read_identity) = guest_shared_backing_identity(address, byte_count) else {
        return false;
    };
    if range.physical_identity.backing_id != read_identity.backing_id {
        return false;
    }
    guest_ranges_overlap(
        range.physical_identity.backing_offset,
        range.byte_count,
        read_identity.backing_offset,
        byte_count,
    )
}

impl MetalPresenter {
    pub(crate) fn publish_guest_batch_completion(&mut self, batch: &GuestCommandBatch) {
        for &sequence in &batch.completion_sequences {
            self.publish_guest_submission_completion(sequence);
        }
    }
}

pub(crate) fn completed_command_buffer_gpu_duration(command_buffer: &CommandBuffer) -> Duration {
    // Metal exposes GPUStartTime/GPUEndTime only after a command buffer completes. The metal-rs
    // version used here does not wrap them yet, so query the public MTLCommandBuffer selectors
    // directly. Unlike wall time from commit to retirement, this excludes time waiting behind
    // older command buffers and is therefore suitable for the title bar's GPU utilization.
    let (start, end): (f64, f64) = unsafe {
        (
            msg_send![command_buffer.as_ref(), GPUStartTime],
            msg_send![command_buffer.as_ref(), GPUEndTime],
        )
    };
    if !start.is_finite() || !end.is_finite() || start <= 0.0 || end < start {
        return Duration::ZERO;
    }
    Duration::from_secs_f64(end - start)
}

fn completed_command_buffer_ack_timeline(
    command_buffer: &CommandBuffer,
) -> SharedEventAckTerminalTimes {
    let (kernel_start, kernel_end, gpu_start, gpu_end): (f64, f64, f64, f64) = unsafe {
        (
            msg_send![command_buffer.as_ref(), kernelStartTime],
            msg_send![command_buffer.as_ref(), kernelEndTime],
            msg_send![command_buffer.as_ref(), GPUStartTime],
            msg_send![command_buffer.as_ref(), GPUEndTime],
        )
    };
    let delta = |start: f64, end: f64| {
        if start.is_finite() && end.is_finite() && start > 0.0 && end >= start {
            Duration::from_secs_f64(end - start)
        } else {
            Duration::ZERO
        }
    };
    SharedEventAckTerminalTimes {
        kernel_to_gpu_start: delta(kernel_start, gpu_start),
        gpu_execution: delta(gpu_start, gpu_end),
        gpu_to_kernel_end: delta(gpu_end, kernel_end),
    }
}

pub(crate) fn collect_pending_global_writebacks(
    buffers: &[UploadedGlobalBuffer],
    evaluation: &ShaderEvaluation,
    pending: &mut Vec<PendingGlobalWriteback>,
) -> Result<(), String> {
    if buffers.len() != evaluation.global_memory_bindings.len() {
        return Err("Metal/global buffer binding count mismatch".to_owned());
    }
    for (buffer, binding) in buffers
        .iter()
        .zip(&evaluation.global_memory_bindings)
        .filter(|(buffer, binding)| {
            !buffer.guest_memory && binding.writable && binding.write_back_to_guest
        })
    {
        if binding.base_address < 0x10_000 || binding.byte_count > 128 * 1024 * 1024 {
            return Err(format!(
                "invalid writable global buffer address={:#x} bytes={}",
                binding.base_address, binding.byte_count
            ));
        }
        if pending.iter().any(|writeback| {
            writeback.buffer.as_ptr() == buffer.buffer.as_ptr()
                && writeback.offset == buffer.offset
                && writeback.guest_address == binding.base_address
                && writeback.byte_count == binding.byte_count
        }) {
            continue;
        }
        let byte_count = u64::try_from(binding.byte_count)
            .map_err(|_| "writable global buffer byte count does not fit u64")?;
        // The capture itself claims the destination (plan D2): an upload-time identity is
        // adopted, otherwise the range is resolved and pinned now. Publication then reuses
        // this claim instead of re-deriving identity and epoch after the queue drains.
        let lease = buffer
            .physical_identity
            .map(|identity| GuestRangeLease::adopt(binding.base_address, byte_count, identity))
            .or_else(|| GuestRangeLease::acquire(binding.base_address, byte_count));
        let write_watch = if lease.is_some() {
            None
        } else {
            // SAFETY: shader evaluation accepted this exact range as a live readable/writable
            // guest binding. Guest mappings are 16 KiB aligned, so its outward-aligned host pages
            // cannot straddle an unrelated reservation. Address-space replacement and protection
            // retire this watch before mutating those pages.
            Some(unsafe { CpuWriteWatch::new(binding.base_address, byte_count) }.map_err(
                |error| {
                    format!(
                        "could not pin copied writable global buffer address={:#x} bytes={}: {error}",
                        binding.base_address, binding.byte_count
                    )
                },
            )?)
        };
        let offset = usize::try_from(buffer.offset)
            .map_err(|_| "writable global buffer offset does not fit usize")?;
        let end = offset
            .checked_add(binding.byte_count)
            .ok_or("writable global buffer range overflow")?;
        if u64::try_from(end).map_or(true, |end| end > buffer.buffer.length()) {
            return Err(format!(
                "writable global buffer range exceeds Metal buffer offset={} bytes={} length={}",
                buffer.offset,
                binding.byte_count,
                buffer.buffer.length()
            ));
        }
        // Ordinary shared uploads still contain their packet-time input before commit. GPU-COW
        // destinations do not: their baseline blit is intentionally pending behind a Metal
        // event wait. Retain that immutable source and resolve it only after the consumer batch
        // is terminal, when both source and destination are CPU-readable.
        let (original_bytes, original_version) = if let Some(version) = &buffer.original_version {
            (Vec::new(), Some(version.clone()))
        } else {
            let bytes = super::super::MetalBytes::new(&buffer.buffer)
                .slice(offset, binding.byte_count)
                .ok_or("writable global source range exceeds its Metal buffer")?
                .to_vec();
            (bytes, None)
        };
        pending.push(PendingGlobalWriteback {
            buffer: buffer.buffer.clone(),
            offset: buffer.offset,
            guest_address: binding.base_address,
            byte_count: binding.byte_count,
            lease,
            write_watch,
            original_bytes,
            original_version,
        });
    }
    Ok(())
}

pub(crate) fn collect_shared_event_global_versions(
    buffers: &[UploadedGlobalBuffer],
    evaluation: &ShaderEvaluation,
    pending: &mut Vec<SharedEventGlobalBufferVersion>,
) -> Result<(), String> {
    if buffers.len() != evaluation.global_memory_bindings.len() {
        return Err("Metal/global buffer binding count mismatch".to_owned());
    }
    for (buffer, binding) in buffers
        .iter()
        .zip(&evaluation.global_memory_bindings)
        .filter(|(buffer, binding)| binding.writable && !buffer.guest_memory)
    {
        let byte_count = u64::try_from(binding.byte_count)
            .map_err(|_| "writable global buffer byte count does not fit u64")?;
        let mapping_identity = buffer
            .physical_identity
            .or_else(|| guest_shared_backing_identity(binding.base_address, byte_count));
        let version = SharedEventGlobalBufferVersion {
            buffer: buffer.buffer.clone(),
            offset: buffer.offset,
            guest_address: binding.base_address,
            byte_count,
            mapping_identity,
        };
        if let Some(existing) = pending.iter_mut().find(|existing| {
            existing.byte_count == version.byte_count
                && match (existing.mapping_identity, version.mapping_identity) {
                    (Some(existing), Some(version)) => existing == version,
                    (None, None) => existing.guest_address == version.guest_address,
                    _ => false,
                }
        }) {
            *existing = version;
        } else {
            pending.push(version);
        }
    }
    Ok(())
}

pub(crate) fn collect_pending_guest_write_ranges(
    buffers: &[UploadedGlobalBuffer],
    evaluation: &ShaderEvaluation,
    pending: &mut Vec<PendingGuestWriteRange>,
) -> Result<(), String> {
    if buffers.len() != evaluation.global_memory_bindings.len() {
        return Err("Metal/global buffer binding count mismatch".to_owned());
    }
    for (buffer, binding) in buffers
        .iter()
        .zip(&evaluation.global_memory_bindings)
        .filter(|(buffer, binding)| binding.writable && buffer.guest_memory)
    {
        let byte_count = u64::try_from(binding.byte_count)
            .map_err(|_| "writable guest buffer byte count does not fit u64")?;
        let physical_identity = buffer.physical_identity.ok_or_else(|| {
                format!(
                    "zero-copy writable guest buffer has no captured physical identity address={:#x} bytes={byte_count}",
                    binding.base_address
                )
            })?;
        let range = PendingGuestWriteRange {
            address: binding.base_address,
            byte_count,
            buffer: buffer.buffer.clone(),
            offset: buffer.offset,
            physical_identity,
        };
        append_or_replace_guest_write_range(pending, range);
    }
    Ok(())
}

pub(crate) fn append_unique_global_writebacks(
    destination: &mut Vec<PendingGlobalWriteback>,
    source: Vec<PendingGlobalWriteback>,
) {
    for writeback in source {
        let duplicate = destination.iter().any(|existing| {
            existing.buffer.as_ptr() == writeback.buffer.as_ptr()
                && existing.offset == writeback.offset
                && existing.guest_address == writeback.guest_address
                && existing.byte_count == writeback.byte_count
        });
        if !duplicate {
            destination.push(writeback);
        }
    }
}

pub(crate) fn append_unique_guest_write_ranges(
    destination: &mut Vec<PendingGuestWriteRange>,
    source: Vec<PendingGuestWriteRange>,
) {
    for range in source {
        append_or_replace_guest_write_range(destination, range);
    }
}

fn append_or_replace_guest_write_range(
    destination: &mut Vec<PendingGuestWriteRange>,
    range: PendingGuestWriteRange,
) {
    if let Some(existing) = destination.iter_mut().find(|existing| {
        existing.physical_identity == range.physical_identity
            && existing.byte_count == range.byte_count
    }) {
        *existing = range;
    } else {
        destination.push(range);
    }
}

pub(crate) fn write_back_pending_globals(
    pending: &[PendingGlobalWriteback],
) -> Result<Vec<&PendingGlobalWriteback>, String> {
    let mut applied = Vec::with_capacity(pending.len());
    for writeback in pending {
        let original_bytes = pending_global_original_bytes(writeback)?;
        let offset = usize::try_from(writeback.offset)
            .map_err(|_| "writable global buffer offset does not fit usize")?;
        let end = offset
            .checked_add(writeback.byte_count)
            .ok_or("writable global buffer range overflow")?;
        if u64::try_from(end).map_or(true, |end| end > writeback.buffer.length()) {
            return Err(format!(
                "writable global buffer range exceeds Metal buffer offset={} bytes={} length={}",
                writeback.offset,
                writeback.byte_count,
                writeback.buffer.length()
            ));
        }
        let current_bytes = super::super::MetalBytes::new(&writeback.buffer)
            .slice(offset, writeback.byte_count)
            .ok_or("writable global writeback range exceeds its Metal buffer")?;
        let outcome = if let Some(lease) = writeback.lease.as_ref() {
            let byte_count = u64::try_from(writeback.byte_count)
                .map_err(|_| "writable global buffer byte count does not fit u64")?;
            if !lease.live_va_matches() {
                trace_discarded_stale_global_writeback(
                    writeback.guest_address,
                    writeback.byte_count,
                    "before completion",
                    lease.identity(),
                    guest_shared_backing_identity(writeback.guest_address, byte_count),
                );
                continue;
            }
            write_changed_global_ranges_leased(writeback, current_bytes, original_bytes, lease)?
        } else if let Some(write_watch) = writeback.write_watch.as_ref() {
            match write_watch.with_live_mapping(|| {
                write_changed_global_ranges_direct(writeback, current_bytes, original_bytes)
            }) {
                Ok(result) => {
                    result?;
                    GlobalWritebackOutcome::Applied
                }
                Err(error) if error.kind() == ErrorKind::NotFound => {
                    trace_discarded_stale_anonymous_global_writeback(
                        writeback.guest_address,
                        writeback.byte_count,
                        &error,
                    );
                    GlobalWritebackOutcome::DiscardedStale
                }
                Err(error) => {
                    return Err(format!(
                        "copied writable global mapping could not be pinned for completion address={:#x} bytes={}: {error}",
                        writeback.guest_address, writeback.byte_count
                    ));
                }
            }
        } else {
            #[cfg(test)]
            {
                write_changed_global_ranges_direct(writeback, current_bytes, original_bytes)?;
                GlobalWritebackOutcome::Applied
            }
            #[cfg(not(test))]
            {
                return Err(format!(
                    "writable global buffer has neither backing identity nor mapping watch address={:#x} bytes={}",
                    writeback.guest_address, writeback.byte_count
                ));
            }
        };
        if outcome == GlobalWritebackOutcome::Applied {
            applied.push(writeback);
        }
    }
    Ok(applied)
}

fn pending_global_original_bytes(writeback: &PendingGlobalWriteback) -> Result<&[u8], String> {
    let Some(version) = &writeback.original_version else {
        if writeback.original_bytes.len() != writeback.byte_count {
            return Err(format!(
                "writable global buffer snapshot length mismatch bytes={} snapshot={}",
                writeback.byte_count,
                writeback.original_bytes.len(),
            ));
        }
        return Ok(&writeback.original_bytes);
    };
    let offset = usize::try_from(version.offset)
        .map_err(|_| "writable global baseline offset does not fit usize")?;
    let end = offset
        .checked_add(writeback.byte_count)
        .ok_or("writable global baseline range overflow")?;
    if u64::try_from(end).map_or(true, |end| end > version.buffer.length()) {
        return Err(format!(
            "writable global baseline exceeds Metal buffer offset={} bytes={} length={}",
            version.offset,
            writeback.byte_count,
            version.buffer.length(),
        ));
    }
    // The bounds above cover the exact slice, and callers invoke this only after the consumer
    // command buffer (baseline copy + compute write) reached a terminal state.
    super::super::MetalBytes::new(&version.buffer)
        .slice(offset, writeback.byte_count)
        .ok_or_else(|| "writable global baseline exceeds its Metal buffer".to_owned())
}

fn write_changed_global_ranges_leased(
    writeback: &PendingGlobalWriteback,
    current_bytes: &[u8],
    original_bytes: &[u8],
    lease: &GuestRangeLease,
) -> Result<GlobalWritebackOutcome, String> {
    let mut cursor = 0;
    while cursor < current_bytes.len() {
        while cursor < current_bytes.len() && current_bytes[cursor] == original_bytes[cursor] {
            cursor += 1;
        }
        let start = cursor;
        while cursor < current_bytes.len() && current_bytes[cursor] != original_bytes[cursor] {
            cursor += 1;
        }
        if start == cursor {
            continue;
        }
        let run_offset = u64::try_from(start)
            .map_err(|_| "writable global buffer changed offset does not fit u64")?;
        let address = writeback
            .guest_address
            .checked_add(run_offset)
            .ok_or("writable global buffer changed address overflow")?;
        let byte_count = u64::try_from(cursor - start)
            .map_err(|_| "writable global buffer changed byte count does not fit u64")?;
        let source = &current_bytes[start..cursor];
        let baseline = &original_bytes[start..cursor];
        match lease.publish_run_at_live_va_if_baseline(run_offset, baseline, source) {
            LeasePublishOutcome::AppliedAtLiveVa => {
                journal_guest_write(GuestWriteSite::GlobalWriteback, address, byte_count, source);
            }
            LeasePublishOutcome::AppliedByIdentity => {
                return Err(format!(
                    "live-VA-only global writeback unexpectedly published by identity address={address:#x} bytes={}",
                    cursor - start,
                ));
            }
            LeasePublishOutcome::GuestModified => {
                trace_kept_guest_bytes_over_writeback(address, cursor - start, "leased");
            }
            LeasePublishOutcome::Revoked => {
                trace_discarded_stale_global_writeback(
                    address,
                    cursor - start,
                    "lease revoked",
                    lease.identity(),
                    guest_shared_backing_identity(address, byte_count),
                );
                return Ok(GlobalWritebackOutcome::DiscardedStale);
            }
            LeasePublishOutcome::WriteFailed => {
                return Err(format!(
                    "writable global buffer backing write failed address={address:#x} bytes={} identity={:?}",
                    cursor - start,
                    lease.identity(),
                ));
            }
        }
    }
    Ok(GlobalWritebackOutcome::Applied)
}

fn trace_discarded_stale_global_writeback(
    address: u64,
    byte_count: usize,
    stage: &str,
    expected: GuestSharedMappingIdentity,
    current: Option<GuestSharedMappingIdentity>,
) {
    let sequence = DISCARDED_STALE_GLOBAL_WRITEBACKS.fetch_add(1, Ordering::Relaxed) + 1;
    if sequence <= 16 || sequence.is_power_of_two() {
        eprintln!(
            "[REMu][METAL] discarded stale writable global writeback sequence={sequence} mapping changed {stage} address={address:#x} bytes={byte_count} expected={expected:?} current={current:?}"
        );
    }
}

fn trace_discarded_stale_anonymous_global_writeback(
    address: u64,
    byte_count: usize,
    error: &std::io::Error,
) {
    let sequence = DISCARDED_STALE_GLOBAL_WRITEBACKS.fetch_add(1, Ordering::Relaxed) + 1;
    if sequence <= 16 || sequence.is_power_of_two() {
        eprintln!(
            "[REMu][METAL] discarded stale writable global writeback sequence={sequence} anonymous mapping changed before completion address={address:#x} bytes={byte_count}: {error}"
        );
    }
}

fn write_changed_global_ranges_direct(
    writeback: &PendingGlobalWriteback,
    current_bytes: &[u8],
    original_bytes: &[u8],
) -> Result<(), String> {
    let destination = usize::try_from(writeback.guest_address)
        .map_err(|_| "writable global buffer address does not fit usize")?;
    let mut cursor = 0;
    while cursor < current_bytes.len() {
        while cursor < current_bytes.len() && current_bytes[cursor] == original_bytes[cursor] {
            cursor += 1;
        }
        let start = cursor;
        while cursor < current_bytes.len() && current_bytes[cursor] != original_bytes[cursor] {
            cursor += 1;
        }
        if start == cursor {
            continue;
        }
        // SAFETY: Production holds the mapping's CpuWriteWatch registry lock across this read and
        // copy. Unit tests keep their local destination allocation alive for the whole call.
        let guest_now = unsafe {
            std::slice::from_raw_parts((destination as *const u8).add(start), cursor - start)
        };
        if guest_now != &original_bytes[start..cursor] {
            trace_kept_guest_bytes_over_writeback(
                writeback.guest_address.wrapping_add(start as u64),
                cursor - start,
                "anonymous",
            );
            continue;
        }
        // SAFETY: Production holds the mapping's CpuWriteWatch registry lock across this copy.
        // Unit tests keep their local destination allocation alive for the duration of the call.
        unsafe {
            std::ptr::copy_nonoverlapping(
                current_bytes[start..cursor].as_ptr(),
                (destination as *mut u8).add(start),
                cursor - start,
            );
        }
        journal_guest_write(
            GuestWriteSite::GlobalWritebackDirect,
            writeback.guest_address.wrapping_add(start as u64),
            (cursor - start) as u64,
            &current_bytes[start..cursor],
        );
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{GuestBatchWork, describe_guest_batch};

    #[test]
    fn batch_diagnostics_retain_indirect_geometry_and_dispatcher_cost() {
        let description = describe_guest_batch(&[
            GuestBatchWork::Compute {
                sequence: 17,
                shader: 0x8008_2200_00,
                source: remu_gpu::ComputeDispatchSource::Indirect {
                    dimensions_address: 0x807c_4b00_00,
                },
                grid: remu_gpu::ComputeGrid::Threadgroups([65_535, 4, 1]),
                threads_per_group: [8, 8, 1],
                msl_bytes: 108_553,
                dispatcher: true,
            },
            GuestBatchWork::Draw {
                sequence: 23,
                indexed: true,
                indirect_arguments_address: Some(0x807d_3600_28),
                export: 0x80dc_3b00_00,
                pixel: 0x80dc_4000_00,
                index_count: 36,
                instance_count: 1,
                target: 0x8020_0000_00,
                extent: [1_920, 1_080],
                export_msl_bytes: 83_048,
                pixel_msl_bytes: 447_366,
                export_dispatcher: false,
                pixel_dispatcher: true,
            },
        ]);

        assert!(description.contains(
            "source=Indirect { dimensions_address: 551841103872 },grid=Threadgroups([65535, 4, 1]),local=[8, 8, 1]"
        ));
        assert!(description.contains("cs=0x8008220000/108553B/dispatcher=true"));
        assert!(description.contains("draw(seq=23,indexed=true,source=indirect@0x807d360028"));
        assert!(description.contains(
            "ps=0x80dc400000/447366B/dispatcher=true,indices=36,instances=1,target=0x8020000000/[1920, 1080]"
        ));
    }
}
