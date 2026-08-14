// SPDX-License-Identifier: GPL-2.0-or-later

//! Ordered guest GPU events and the command payloads they carry.

use crate::{ComputeDispatch, GpuHostImage, HostDialogRequest, HostToastRequest, IndexedDraw};
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

const GPU_MEMORY_COPY_SNAPSHOT_BUDGET_BYTES: usize = 256 * 1024 * 1024;

static GPU_MEMORY_COPY_SNAPSHOT_RETAINED_BYTES: AtomicUsize = AtomicUsize::new(0);

static GPU_MEMORY_COPY_SNAPSHOT_PEAK_BYTES: AtomicUsize = AtomicUsize::new(0);

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum GpuEvent {
    /// One complete guest command-buffer submission and the hardware queue that owns it.
    ///
    /// Events inside a submission are strictly ordered. A backend may interleave work from
    /// different queue IDs, but it must never execute a later event from this submission while
    /// its front event is blocked. `QueueSubmission` must not be nested.
    QueueSubmission {
        queue_id: u64,
        events: Vec<GpuEvent>,
    },
    DisplayRegistered {
        width: u32,
        height: u32,
        pitch: u32,
        pixel_format: u64,
        tiling_mode: u32,
        buffers: Vec<u64>,
    },
    /// A host-decoded installed-title image displayed while the system splash is visible.
    ShowHostSplash(GpuHostImage),
    /// Removes the system-owned splash, revealing the newest guest scanout (or black).
    HideHostSplash,
    /// A modal host dialog the guest common-dialog ABIs poll until the user dismisses it.
    ShowHostDialog(HostDialogRequest),
    /// Forces any visible modal dialog closed (guest `Close` / `Terminate`).
    HideHostDialog {
        sequence: u64,
    },
    /// A non-modal toast (trophy unlock, notification) that does not block dialog polling.
    ShowHostToast(HostToastRequest),
    Flip {
        address: u64,
        buffer_index: u32,
        flip_argument: u64,
        /// Exact guest submission containing the in-stream flip packet.
        ///
        /// Standalone VideoOut flips are host events and deliberately carry no submission.
        submission_sequence: Option<u64>,
        width: u32,
        height: u32,
        pitch: u32,
        pixel_format: u64,
        tiling_mode: u32,
    },
    /// A Gen5 `WAIT_FLIP_DONE` command-processor dependency.
    ///
    /// The backend binds an unbound wait to the newest flip it has already accepted for this
    /// exact VideoOut buffer. Keeping that generation in the requeued event prevents a newer flip
    /// from extending an older wait while its queue front is parked. `None` therefore means
    /// "bind on first execution", not "wait for every future flip".
    WaitFlipDone(GpuFlipWait),
    /// A `PFP_SYNC_ME` command-processor dependency.
    ///
    /// Following parser-front-end work, especially indirect argument fetches, must not overtake
    /// preceding micro-engine work which can produce those arguments.
    CommandProcessorBarrier,
    PipelineBarrier(GpuPipelineBarrier),
    /// CB/DB writeback-invalidate `EVENT_WRITE` (`drawcommandbuffer.h` EventType 0x16/0x2A/0x2C/0x2E/0x31).
    ///
    /// SDK `sync.cpp` emits these before end-of-pipe / acquire paths; they are the only render-backend
    /// flush mechanism. On Metal, ending the open render pass stores attachments (TBDR writeback).
    RenderBackendFlush(GpuRenderBackendFlush),
    MemoryAcquire(GpuMemoryAcquire),
    MemoryCopy(GpuMemoryCopy),
    /// `gpuSyncDma`: DMA with `DmaDataDst::kNowhere` / size 0 that waits for prior DMAs and blocks the CP.
    DmaSyncBarrier,
    MemoryWrite(GpuMemoryWrite),
    MemoryRelease(GpuMemoryRelease),
    /// A Gen5 `PIXEL_PIPE_STAT_DUMP` / `kOcclusionCounterDump` (event type 0x39).
    ///
    /// The destination is the qword-aligned base of a 16×16-byte render-backend counter block
    /// (`sce::Agc::OcclusionQueryResults` / `Alignment::kOcclusionQueryResults` —
    /// `sdk/target/include/agc/drawcommandbuffer.h:85-86,385-392` and
    /// `sdk/target/include/agc/baselayer.h:67`). The Metal backend publishes real
    /// `MTLVisibilityResultBuffer` sample counts (plus the high validity bit) into that block;
    /// the runtime must not fabricate counter values.
    OcclusionQuery {
        destination: u64,
    },
    /// A decoded wait which has not yet crossed the installed guest GPU event sink.
    ///
    /// [`crate::emit_event`] upgrades waits inside a `QueueSubmission` to
    /// [`GpuEvent::RegisteredMemoryWait`] before the bounded channel send. Keeping this variant
    /// separate lets command decoding and backend unit tests use the pure packet condition without
    /// fabricating a process-global ticket.
    MemoryWait(GpuMemoryWait),
    /// A wait registered synchronously when its owning queue submission was published.
    RegisteredMemoryWait {
        id: GpuWaitId,
        wait: GpuMemoryWait,
    },
    ComputeDispatch(Box<ComputeDispatch>),
    IndexedDraw(Arc<IndexedDraw>),
    /// An ordered CPU-visible end-of-pipe boundary for one guest graphics submission.
    ///
    /// The backend records this sequence only after all preceding translated GPU work and guest
    /// memory writebacks have completed. The kernel equeue implementation uses the resulting
    /// monotonic value to deliver `sceAgcDriverAddEqEvent` completion notifications.
    SubmissionBoundary(u64),
    Shutdown,
}

/// One queue-ordered request to stop rendering until a VideoOut buffer is reusable.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct GpuFlipWait {
    pub address: u64,
    pub buffer_index: u32,
    pub required_flip_sequence: Option<u64>,
}

/// One display-buffer flip whose presentation command reached a terminal backend state.
///
/// The address, buffer index and guest argument form the VideoOut-visible identity. A title may
/// reuse an argument, so the completion tracker retains duplicates and the runtime consumes them
/// one-for-one in submission order.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct GpuFlipCompletion {
    pub address: u64,
    pub buffer_index: u32,
    pub flip_argument: u64,
}

/// A guest pipeline-stage completion boundary emitted by a PM4 partial-flush event.
///
/// Backends may use a lighter native primitive than a CPU wait, but following GPU work must not
/// overtake work from the named stage.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct GpuPipelineBarrier {
    pub wait_for: GpuPipelineStage,
}

/// One of the CB/DB writeback-invalidate events from SDK `EventType`.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum GpuRenderBackendFlush {
    /// `EventType::kCbDbDataWritebackInvalidate` (0x16).
    CbDbData,
    /// `EventType::kDbDataWritebackInvalidate` (0x2A).
    DbData,
    /// `EventType::kDbMetadataWritebackInvalidate` (0x2C).
    DbMetadata,
    /// `EventType::kCbMetadataWritebackInvalidate` (0x2E).
    CbMetadata,
    /// `EventType::kCbDataWritebackInvalidate` (0x31).
    CbData,
}

impl GpuRenderBackendFlush {
    #[must_use]
    pub const fn from_event_type(event_type: u32) -> Option<Self> {
        match event_type {
            0x16 => Some(Self::CbDbData),
            0x2A => Some(Self::DbData),
            0x2C => Some(Self::DbMetadata),
            0x2E => Some(Self::CbMetadata),
            0x31 => Some(Self::CbData),
            _ => None,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum GpuPipelineStage {
    ComputeShader,
    VertexShader,
    PixelShader,
}

/// A decoded Gen5 `ACQUIRE_MEM` packet in guest byte-address form.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct GpuMemoryAcquire {
    pub engine: u32,
    pub cb_db_control: u32,
    pub base_address: u64,
    pub size_bytes: u64,
    pub poll_interval: u32,
    pub gcr_control: u32,
}

/// An ordered command-processor byte copy emitted by a Gen5 `DMA_DATA` packet.
#[derive(Clone, Eq, PartialEq)]
pub struct GpuMemoryCopy {
    pub source_address: u64,
    pub destination_address: u64,
    pub byte_count: u64,
    pub source_selection: u32,
    pub destination_selection: u32,
    /// Immutable CPU-source bytes observed while decoding this packet.
    ///
    /// GPU-authored sources ignore this snapshot after their ordered producer completes.
    pub source_snapshot: Option<GpuMemoryCopySnapshot>,
}

impl std::fmt::Debug for GpuMemoryCopy {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("GpuMemoryCopy")
            .field("source_address", &self.source_address)
            .field("destination_address", &self.destination_address)
            .field("byte_count", &self.byte_count)
            .field("source_selection", &self.source_selection)
            .field("destination_selection", &self.destination_selection)
            .field(
                "source_snapshot_bytes",
                &self.source_snapshot.as_ref().map(|snapshot| snapshot.len()),
            )
            .finish()
    }
}

#[derive(Debug, Eq, PartialEq)]
struct GpuMemoryCopySnapshotAllocation {
    bytes: Arc<[u8]>,
}

impl Drop for GpuMemoryCopySnapshotAllocation {
    fn drop(&mut self) {
        GPU_MEMORY_COPY_SNAPSHOT_RETAINED_BYTES.fetch_sub(self.bytes.len(), Ordering::AcqRel);
    }
}

/// A globally budgeted immutable DMA source allocation.
#[derive(Clone, Eq, PartialEq)]
pub struct GpuMemoryCopySnapshot(Arc<GpuMemoryCopySnapshotAllocation>);

impl GpuMemoryCopySnapshot {
    /// Retains an existing packet-time allocation without copying it.
    pub fn try_new(bytes: Arc<[u8]>) -> Result<Self, GpuMemoryCopySnapshotBudgetExceeded> {
        let byte_count = bytes.len();
        let retained = GPU_MEMORY_COPY_SNAPSHOT_RETAINED_BYTES
            .fetch_update(Ordering::AcqRel, Ordering::Acquire, |retained| {
                retained
                    .checked_add(byte_count)
                    .filter(|total| *total <= GPU_MEMORY_COPY_SNAPSHOT_BUDGET_BYTES)
            })
            .map_err(|retained| GpuMemoryCopySnapshotBudgetExceeded {
                requested_bytes: retained.saturating_add(byte_count),
                limit_bytes: GPU_MEMORY_COPY_SNAPSHOT_BUDGET_BYTES,
            })?
            .saturating_add(byte_count);
        GPU_MEMORY_COPY_SNAPSHOT_PEAK_BYTES.fetch_max(retained, Ordering::Relaxed);
        Ok(Self(Arc::new(GpuMemoryCopySnapshotAllocation { bytes })))
    }

    #[must_use]
    pub fn len(&self) -> usize {
        self.0.bytes.len()
    }

    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.0.bytes.is_empty()
    }

    #[must_use]
    pub fn as_ptr(&self) -> *const u8 {
        self.0.bytes.as_ptr()
    }
}

impl std::ops::Deref for GpuMemoryCopySnapshot {
    type Target = [u8];

    fn deref(&self) -> &Self::Target {
        &self.0.bytes
    }
}

impl std::fmt::Debug for GpuMemoryCopySnapshot {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("GpuMemoryCopySnapshot")
            .field("bytes", &self.len())
            .finish()
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct GpuMemoryCopySnapshotBudgetExceeded {
    pub requested_bytes: usize,
    pub limit_bytes: usize,
}

/// Returns current, peak, and maximum retained DMA source snapshot bytes.
#[must_use]
pub fn gpu_memory_copy_snapshot_statistics() -> (usize, usize, usize) {
    (
        GPU_MEMORY_COPY_SNAPSHOT_RETAINED_BYTES.load(Ordering::Acquire),
        GPU_MEMORY_COPY_SNAPSHOT_PEAK_BYTES.load(Ordering::Relaxed),
        GPU_MEMORY_COPY_SNAPSHOT_BUDGET_BYTES,
    )
}

#[cfg(test)]
mod memory_copy_snapshot_tests {
    use super::{GpuMemoryCopySnapshot, gpu_memory_copy_snapshot_statistics};
    use std::sync::Arc;

    #[test]
    fn snapshot_budget_accounts_one_shared_allocation_until_its_last_owner_drops() {
        let before = gpu_memory_copy_snapshot_statistics().0;
        let snapshot = GpuMemoryCopySnapshot::try_new(Arc::<[u8]>::from([1, 2, 3, 4]))
            .expect("small snapshot fits global budget");
        assert_eq!(gpu_memory_copy_snapshot_statistics().0, before + 4);
        let clone = snapshot.clone();
        assert_eq!(
            gpu_memory_copy_snapshot_statistics().0,
            before + 4,
            "cloning an event must not charge shared bytes twice"
        );
        drop(snapshot);
        assert_eq!(gpu_memory_copy_snapshot_statistics().0, before + 4);
        drop(clone);
        assert_eq!(gpu_memory_copy_snapshot_statistics().0, before);
    }
}

/// Gen5 occlusion / pixel-pipe dump result blocks contain sixteen 16-byte render-backend pairs.
///
/// SDK: `sce::Agc::OcclusionQueryResults::m_results[16]` (`drawcommandbuffer.h:392`) and
/// `Alignment::kOcclusionQueryResults` (`baselayer.h:67`).
pub const PIXEL_PIPE_STAT_COUNTER_PAIRS: u64 = 16;

/// High bit set on every published occlusion / pixel-pipe counter qword once the GPU write has
/// retired. Native titles (and Unreal's blocking render-query path) spin until this bit appears.
pub const PIXEL_PIPE_STAT_VALID: u64 = 1 << 63;

/// Gen5 `CachePolicy` for CP label / wait packets (`drawcommandbuffer.h`).
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(u8)]
pub enum GpuCachePolicy {
    Lru = 0,
    Stream = 1,
    Noalloc = 2,
    Bypass = 3,
}

impl GpuCachePolicy {
    #[must_use]
    pub const fn from_u32(value: u32) -> Option<Self> {
        match value {
            0 => Some(Self::Lru),
            1 => Some(Self::Stream),
            2 => Some(Self::Noalloc),
            3 => Some(Self::Bypass),
            _ => None,
        }
    }
}

/// `EndOfPipeCbDbAction` values carried in `RELEASE_MEM` (`commandbuffer.h`).
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(u8)]
pub enum GpuEndOfPipeCbDbAction {
    WritebackCbDbData = 0x04,
    WritebackInvalidateCbDbData = 0x14,
    None = 0x28,
    WritebackInvalidateDbData = 0x2B,
    WritebackInvalidateCbData = 0x2D,
}

impl GpuEndOfPipeCbDbAction {
    #[must_use]
    pub const fn from_u32(value: u32) -> Option<Self> {
        match value {
            0x04 => Some(Self::WritebackCbDbData),
            0x14 => Some(Self::WritebackInvalidateCbDbData),
            0x28 => Some(Self::None),
            0x2B => Some(Self::WritebackInvalidateDbData),
            0x2D => Some(Self::WritebackInvalidateCbData),
            _ => None,
        }
    }

    /// SDK: any non-`kNone` action flushes CB/DB data caches before the label write.
    #[must_use]
    pub const fn flushes_render_backend(self) -> bool {
        !matches!(self, Self::None)
    }
}

/// `WaitOperation` engine select from `WAIT_REG_MEM` (`drawcommandbuffer.h`).
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(u8)]
pub enum GpuWaitOperation {
    /// PFP waits — parser-front work cannot overtake the compare.
    WaitPfp = 1,
    /// ME waits — PFP may run ahead of the stalled micro-engine.
    WaitMe = 0,
    /// Conditional ME wait (`CondWriteSpace::kScratch`).
    WaitCondMe = 4,
}

impl GpuWaitOperation {
    #[must_use]
    pub const fn from_u32(value: u32) -> Option<Self> {
        match value {
            0 => Some(Self::WaitMe),
            1 => Some(Self::WaitPfp),
            4 => Some(Self::WaitCondMe),
            _ => None,
        }
    }

    /// PFP waits must close open encoders before the GPU wait (parser cannot skip ahead).
    #[must_use]
    pub const fn stalls_parser(self) -> bool {
        matches!(self, Self::WaitPfp)
    }
}

/// An ordered CPU-visible label write produced by a Gen5 end-of-pipe packet.
///
/// Fields beyond address/value come from SDK `queueEndOfPipeAction` / `RELEASE_MEM`
/// (`sync.cpp` `gpuSyncEvent` / `gpuSyncPostProducer`): CB/DB action, GCR control, and the
/// cache policy that distinguishes CPU-visible (`kBypass`/`kNoalloc`) from GPU-visible (`kLru`)
/// labels.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct GpuMemoryRelease {
    pub address: u64,
    pub value: GpuMemoryReleaseValue,
    /// `EndOfPipeCbDbAction` byte from the packet.
    pub action: u32,
    /// `GcrCntl` / acquire GCR bits packed beside data selection.
    pub gcr_control: u32,
    /// `CachePolicy` selecting CPU vs GPU label visibility.
    pub cache_policy: u32,
    /// `QueuedInterrupt` selector (decoded; interrupt delivery is a separate surface).
    pub interrupt: u32,
}

impl GpuMemoryRelease {
    /// Label-only release with SDK `kNone` CB/DB action and GPU-visible LRU policy.
    #[must_use]
    pub const fn label(address: u64, value: GpuMemoryReleaseValue) -> Self {
        Self {
            address,
            value,
            action: GpuEndOfPipeCbDbAction::None as u32,
            gcr_control: 0,
            cache_policy: GpuCachePolicy::Lru as u32,
            interrupt: 0,
        }
    }

    /// SDK: Bypass/Noalloc label stores are CPU-visible; LRU stores are GPU-visible (`sync.cpp`).
    #[must_use]
    pub const fn cpu_visible(self) -> bool {
        self.cache_policy == GpuCachePolicy::Bypass as u32
            || self.cache_policy == GpuCachePolicy::Noalloc as u32
    }

    #[must_use]
    pub const fn flushes_render_backend(self) -> bool {
        match GpuEndOfPipeCbDbAction::from_u32(self.action) {
            Some(action) => action.flushes_render_backend(),
            None => self.action != GpuEndOfPipeCbDbAction::None as u32,
        }
    }
}

/// An ordered command-processor write of one or more dwords to guest memory.
///
/// `WRITE_DATA` can either advance the destination for every dword or repeatedly update one
/// address. The latter is used for register-like labels, while Unity's submit-done fence uses an
/// incrementing memory destination.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct GpuMemoryWrite {
    pub address: u64,
    pub values: Vec<u32>,
    pub increment_address: bool,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum GpuMemoryReleaseValue {
    Immediate32(u32),
    Immediate64(u64),
    Timestamp,
}

/// An ordered Gen5 command-processor wait on a guest-memory label.
///
/// `operation` is the SDK `WaitOperation` (PFP vs ME engine select). Ignoring it collapses every
/// wait to one engine and is why producerless suspensions mis-order relative to PFP fetches.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct GpuMemoryWait {
    pub address: u64,
    pub reference: u64,
    pub mask: u64,
    pub compare_function: u32,
    pub is_64_bit: bool,
    /// `WaitOperation` including the mandatory PFP/ME engine select.
    pub operation: u32,
    /// `CachePolicy` for the wait probe (CPU labels use Noalloc; GPU labels use LRU).
    pub cache_policy: u32,
}

impl GpuMemoryWait {
    #[must_use]
    pub const fn memory(
        address: u64,
        reference: u64,
        mask: u64,
        compare_function: u32,
        is_64_bit: bool,
    ) -> Self {
        Self {
            address,
            reference,
            mask,
            compare_function,
            is_64_bit,
            operation: GpuWaitOperation::WaitPfp as u32,
            cache_policy: GpuCachePolicy::Lru as u32,
        }
    }

    #[must_use]
    pub const fn wait_operation(self) -> Option<GpuWaitOperation> {
        GpuWaitOperation::from_u32(self.operation)
    }
}

/// Process-unique identity for one submitted guest-memory wait.
///
/// A label address can be returned to the guest allocator and reused for another frame. The ID
/// keeps those logical generations distinct: a producer may latch only wait IDs which were
/// registered before that producer retired.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct GpuWaitId(pub(crate) u64);

impl GpuMemoryWait {
    #[must_use]
    pub const fn is_satisfied_by(self, value: u64) -> bool {
        let value = value & self.mask;
        match self.compare_function {
            0 => true,
            1 => value < self.reference,
            2 => value <= self.reference,
            3 => value == self.reference,
            4 => value != self.reference,
            5 => value >= self.reference,
            6 => value > self.reference,
            _ => true,
        }
    }
}

impl GpuMemoryAcquire {
    const GLI_INVALIDATE_MASK: u32 = 0x3;
    const GL1_RANGE_SHIFT: u32 = 2;
    const GL1_RANGE_MASK: u32 = 0x3;
    const GLM_INVALIDATE: u32 = 1 << 5;
    const GLK_INVALIDATE: u32 = 1 << 7;
    const GLV_INVALIDATE: u32 = 1 << 8;
    const GL1_INVALIDATE: u32 = 1 << 9;
    const GL2_RANGE_SHIFT: u32 = 11;
    const GL2_RANGE_MASK: u32 = 0x3;
    const GL2_DISCARD: u32 = 1 << 13;
    const GL2_INVALIDATE: u32 = 1 << 14;

    /// Returns whether this packet invalidates host-cached guest resources.
    #[must_use]
    pub const fn invalidates_guest_resources(self) -> bool {
        self.gcr_control
            & (Self::GLI_INVALIDATE_MASK
                | Self::GLM_INVALIDATE
                | Self::GLK_INVALIDATE
                | Self::GLV_INVALIDATE
                | Self::GL1_INVALIDATE
                | Self::GL2_DISCARD
                | Self::GL2_INVALIDATE)
            != 0
    }

    /// Returns whether the packet's effective invalidation scope covers all guest memory.
    #[must_use]
    pub const fn covers_all_guest_memory(self) -> bool {
        self.size_bytes == 0
            || self.gcr_control & Self::GLI_INVALIDATE_MASK == 1
            || (self.gcr_control
                & (Self::GLM_INVALIDATE
                    | Self::GLK_INVALIDATE
                    | Self::GLV_INVALIDATE
                    | Self::GL1_INVALIDATE)
                != 0
                && (self.gcr_control >> Self::GL1_RANGE_SHIFT) & Self::GL1_RANGE_MASK == 0)
            || (self.gcr_control & (Self::GL2_DISCARD | Self::GL2_INVALIDATE) != 0
                && (self.gcr_control >> Self::GL2_RANGE_SHIFT) & Self::GL2_RANGE_MASK == 0)
    }

    /// Returns whether a guest resource intersects the packet's effective invalidation scope.
    #[must_use]
    pub const fn overlaps(self, address: u64, byte_count: u64) -> bool {
        if self.covers_all_guest_memory() {
            return true;
        }
        let resource_end = address.saturating_add(byte_count);
        let acquire_end = self.base_address.saturating_add(self.size_bytes);
        address < acquire_end && self.base_address < resource_end
    }
}
