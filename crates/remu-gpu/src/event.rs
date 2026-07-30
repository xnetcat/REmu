// SPDX-License-Identifier: GPL-2.0-or-later

//! Ordered guest GPU events and the command payloads they carry.

use crate::{ComputeDispatch, GpuHostImage, IndexedDraw};

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
    Flip {
        address: u64,
        buffer_index: u32,
        flip_argument: u64,
        width: u32,
        height: u32,
        pitch: u32,
        pixel_format: u64,
        tiling_mode: u32,
    },
    /// A `PFP_SYNC_ME` command-processor dependency.
    ///
    /// Following parser-front-end work, especially indirect argument fetches, must not overtake
    /// preceding micro-engine work which can produce those arguments.
    CommandProcessorBarrier,
    PipelineBarrier(GpuPipelineBarrier),
    MemoryAcquire(GpuMemoryAcquire),
    MemoryCopy(GpuMemoryCopy),
    MemoryWrite(GpuMemoryWrite),
    MemoryRelease(GpuMemoryRelease),
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
    IndexedDraw(Box<IndexedDraw>),
    /// An ordered CPU-visible end-of-pipe boundary for one guest graphics submission.
    ///
    /// The backend records this sequence only after all preceding translated GPU work and guest
    /// memory writebacks have completed. The kernel equeue implementation uses the resulting
    /// monotonic value to deliver `sceAgcDriverAddEqEvent` completion notifications.
    SubmissionBoundary(u64),
    Shutdown,
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
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct GpuMemoryCopy {
    pub source_address: u64,
    pub destination_address: u64,
    pub byte_count: u64,
    pub source_selection: u32,
    pub destination_selection: u32,
}

/// An ordered CPU-visible label write produced by a Gen5 end-of-pipe packet.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct GpuMemoryRelease {
    pub address: u64,
    pub value: GpuMemoryReleaseValue,
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
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct GpuMemoryWait {
    pub address: u64,
    pub reference: u64,
    pub mask: u64,
    pub compare_function: u32,
    pub is_64_bit: bool,
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
