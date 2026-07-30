// SPDX-License-Identifier: GPL-2.0-or-later

mod channel;
pub use channel::{
    GPU_EVENT_QUEUE_CAPACITY, GPU_EVENT_SUBMISSION_CHUNK_SIZE, GPU_PENDING_EVENT_CAPACITY,
    acquire_memory, clear_event_sink, emit_event, emit_event_with_admission, gpu_event_channel,
    install_event_sink, pipeline_barrier,
};

mod compute;
pub use compute::{
    COMPUTE_NUM_THREAD_X_REGISTER, COMPUTE_NUM_THREAD_Y_REGISTER, COMPUTE_NUM_THREAD_Z_REGISTER,
    COMPUTE_PGM_HI_REGISTER, COMPUTE_PGM_LO_REGISTER, COMPUTE_PGM_RSRC2_REGISTER,
    COMPUTE_START_X_REGISTER, COMPUTE_START_Y_REGISTER, COMPUTE_START_Z_REGISTER, ComputeDispatch,
    ComputeDispatchDecodeError, ComputeDispatchSource, ComputeGrid, ComputeIndirectDimensions,
    ComputeIndirectPacket, ComputeLaunch, compute_program_address, decode_compute_dispatch_direct,
    decode_compute_dispatch_indirect, decode_compute_launch, decode_compute_system_registers,
};

mod event;
pub use event::{
    GpuEvent, GpuFlipCompletion, GpuMemoryAcquire, GpuMemoryCopy, GpuMemoryCopySnapshot,
    GpuMemoryCopySnapshotBudgetExceeded, GpuMemoryRelease, GpuMemoryReleaseValue, GpuMemoryWait,
    GpuMemoryWrite, GpuPipelineBarrier, GpuPipelineStage, GpuWaitId,
    gpu_memory_copy_snapshot_statistics,
};

mod diagnostics;
pub use diagnostics::{
    PixelPipeQueryCompletionTrace, PixelPipeStatReleaseTrace,
    record_pixel_pipe_query_completion_release_trace, record_pixel_pipe_stat_dump_trace,
    record_pixel_pipe_stat_release_trace, take_pixel_pipe_query_completion_release_trace,
    take_pixel_pipe_stat_release_trace,
};

mod waits;
#[cfg(test)]
use waits::gpu_wait_registry;
pub use waits::{
    pending_gpu_waits_overlapping, record_gpu_wait_producer, retire_gpu_wait, take_gpu_wait_latch,
};

mod counters;
pub use counters::{
    audio_output_frames, audio_progress_age_millis, audio_underrun_frames,
    gpu_completed_submission_ids_snapshot, gpu_completed_submission_sequence, gpu_events_in_flight,
    gpu_flips_pending, gpu_progress_age_millis, gpu_work_items_pending,
    install_gpu_completion_notifier, progress_telemetry_enabled, record_gpu_event_consumed,
    record_gpu_flip_completed, record_gpu_submission_completed, record_gpu_work_item_processed,
    report_audio_frames, set_progress_telemetry_enabled, take_gpu_completed_flips,
    take_gpu_completed_submission_ids,
};

mod resource;
pub use resource::{
    GpuAdapter, GpuColorTarget, GpuHostImage, GpuProbe, GpuProbeError, GpuRegister,
    GpuScanoutTarget, GraphicsApi, IndexedDraw, ShaderSnapshot,
};

mod dialog;
pub use dialog::{
    HostDialogButtons, HostDialogKind, HostDialogRequest, HostDialogResult,
    HostDialogResultCode, HostToastRequest, begin_host_dialog, cancel_host_dialog,
    complete_host_dialog, host_dialog_is_pending, host_dialog_result, reset_host_dialog_state,
};

mod system_ui;
pub use system_ui::{
    SYSTEM_UI_HEIGHT, SYSTEM_UI_WIDTH, SystemUiFramebuffer, rasterize_system_ui,
};

mod pad;
pub use pad::{
    HostPadState, PAD_BUTTON_CIRCLE, PAD_BUTTON_CROSS, PAD_BUTTON_DOWN, PAD_BUTTON_L1,
    PAD_BUTTON_L2, PAD_BUTTON_L3, PAD_BUTTON_LEFT, PAD_BUTTON_OPTIONS, PAD_BUTTON_R1,
    PAD_BUTTON_R2, PAD_BUTTON_R3, PAD_BUTTON_RIGHT, PAD_BUTTON_SQUARE, PAD_BUTTON_TRIANGLE,
    PAD_BUTTON_UP, host_pad_buttons_held, host_pad_state, set_host_pad_state,
};

#[cfg(test)]
mod tests;
