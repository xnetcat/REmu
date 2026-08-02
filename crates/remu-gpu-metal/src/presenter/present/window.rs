// Copyright (C) 2026 REmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

use super::super::completion::{command_buffer_native_state_summary, trace_metal_command_stalls};
use super::super::texture::RETIRED_TEXTURE_QUARANTINE_CAPACITY;
use super::super::{
    OWNED_DEMOTIONS, OWNED_FALLBACK_FULL_RANGE, OWNED_RANGE_SET_COARSENED, OWNED_RECLAIM_DISCARD,
    OWNED_SUBTRACT_TENANTS_SURVIVED, OWNED_UNRESOLVED_SPAN, RESIDENT_WRITEBACK_DISCARDED,
    RESIDENT_WRITEBACK_DISCARDED_BYTES, RESIDENT_WRITEBACK_PUBLISHED,
    RESIDENT_WRITEBACK_PUBLISHED_BYTES, ResidencyState, TEXTURE_QUARANTINE_ADOPTED,
    TEXTURE_QUARANTINE_REJECTED, WRITEBACK_TRIGGERS,
};
use super::{
    EVENT_POLL_INTERVAL, FlushPolicy, GuestBatchFlushReason, MIP_UPLOAD_BYTES, MIP_UPLOAD_COUNT,
    MIP_UPLOAD_NANOSECONDS, MODE27_DECODE_BYTES, MODE27_DECODE_COUNT, MODE27_DECODE_NANOSECONDS,
    MetalPresenter, PIPELINE_COMPILE_COMPLETED, PIPELINE_COMPILE_ENQUEUED, PIPELINE_COMPILE_FAILED,
    PIPELINE_COMPILE_QUEUE_FULL, PIPELINE_COMPUTE_GATES, PIPELINE_DRAW_GATES, PresenterFlow,
    RESIDENT_DEPTH_SAMPLE_HITS, RESIDENT_FLIP_AGE_LABELS, RESIDENT_TEXTURE_ACQUIRE_INVALIDATIONS,
    RESIDENT_WORK_AGE_LABELS, RESIDENT_WRITE_ATTRIBUTION_MISSES, SAMPLED_ALIAS_WRITEBACK_BYTES,
    SAMPLED_ALIAS_WRITEBACK_COUNT,
    SAMPLED_ALIAS_WRITEBACK_NANOSECONDS, SAMPLED_TEXTURE_CACHE_ACQUIRE_INVALIDATIONS,
    SAMPLED_TEXTURE_CACHE_CAPACITY_EVICTIONS, SAMPLED_TEXTURE_CACHE_DIRTY_RELOADS,
    SAMPLED_TEXTURE_CACHE_HITS, SAMPLED_TEXTURE_CACHE_MISSES, TEXTURE_DIRTY_VALIDATION_BYTES,
    TEXTURE_DIRTY_VALIDATION_COUNT, TEXTURE_DIRTY_VALIDATION_NANOSECONDS,
    TEXTURE_DIRTY_VALIDATION_UNCHANGED, TEXTURE_SNAPSHOT_BYTES, TEXTURE_SNAPSHOT_COUNT,
    TEXTURE_SNAPSHOT_NANOSECONDS, TEXTURE_SNAPSHOT_UNSTABLE_COUNT, describe_guest_batch,
    presentation_drawable_size, process_memory_statistics, resident_texture_activity_latest_work,
    resident_texture_activity_stats, resident_texture_backing_overlap_stats,
    resident_texture_lifetime_diagnostics_enabled, resident_texture_ownership_stats,
    take_mode27_work_details, take_texture_dirty_work_details, take_texture_snapshot_work_details,
};
use core_graphics_types::geometry::CGSize;
use metal::{MTLCommandBufferStatus, MTLPixelFormat, MetalLayer};
use objc::runtime::{NO, Object, YES};
use objc::{class, msg_send, sel, sel_impl};
use remu_gpu::{
    GpuEvent, HostPadState, PAD_BUTTON_CIRCLE, PAD_BUTTON_CROSS, PAD_BUTTON_DOWN, PAD_BUTTON_L1,
    PAD_BUTTON_L2, PAD_BUTTON_LEFT, PAD_BUTTON_OPTIONS, PAD_BUTTON_R1, PAD_BUTTON_R2,
    PAD_BUTTON_RIGHT, PAD_BUTTON_SQUARE, PAD_BUTTON_TRIANGLE, PAD_BUTTON_UP, set_host_pad_state,
};
use std::ffi::OsStr;
use std::sync::atomic::Ordering;
use std::sync::mpsc::Receiver;
use std::time::{Duration, Instant};
use winit::application::ApplicationHandler;
use winit::dpi::LogicalSize;
use winit::event::{ElementState, WindowEvent};
use winit::event_loop::{ActiveEventLoop, ControlFlow};
use winit::keyboard::{KeyCode, PhysicalKey};
use winit::raw_window_handle::{HasWindowHandle, RawWindowHandle};
use winit::window::{Window, WindowId};

impl MetalPresenter {
    pub(crate) fn initialize(&mut self, event_loop: &ActiveEventLoop) -> Result<(), String> {
        let window = event_loop
            .create_window(presenter_window_attributes())
            .map_err(|error| format!("could not create the macOS window: {error}"))?;
        self.initialize_window(window)
    }

    pub(crate) fn initialize_window(&mut self, window: Window) -> Result<(), String> {
        let device = self
            .device
            .as_ref()
            .ok_or("Metal presenter resources are unavailable")?;
        let mut layer = MetalLayer::new();
        layer.set_device(device);
        // Default to sRGB for linear float scanouts (Unity RGBA16F). Display-referred Unorm
        // scanouts such as Void Terrarium's RGB10A2 switch the layer to BGRA8Unorm at present
        // time so Metal does not apply a second linear→sRGB encode.
        layer.set_pixel_format(MTLPixelFormat::BGRA8Unorm_sRGB);
        // Presentation uses a fullscreen sampling pass so a high-resolution guest display buffer
        // does not force an equally large host drawable. The layer remains framebuffer-only.
        layer.set_framebuffer_only(true);
        layer.set_presents_with_transaction(false);

        // SAFETY: Winit owns a live NSView on the main thread. NSView retains the CAMetalLayer;
        // this presenter also keeps its owning MetalLayer handle until the event loop exits.
        unsafe {
            let handle = window
                .window_handle()
                .map_err(|error| format!("could not get the AppKit window handle: {error}"))?;
            let RawWindowHandle::AppKit(appkit) = handle.as_raw() else {
                return Err("winit did not create an AppKit window".to_owned());
            };
            let view = appkit.ns_view.as_ptr().cast::<Object>();
            let appkit_window: *mut Object = msg_send![view, window];
            if appkit_window.is_null() {
                return Err("winit's AppKit view is not attached to a window".to_owned());
            }
            // A presenter is rebuilt from the current guest run and has no UI state worth
            // restoring. Titled NSWindows are restorable by default, so opt this window out
            // explicitly before it is ordered onscreen.
            let _: () = msg_send![appkit_window, setRestorable: NO];
            let _: () = msg_send![view, setWantsLayer: YES];
            let _: () = msg_send![view, setLayer: layer.as_mut()];
        }

        let size = window.inner_size();
        let (width, height) = presentation_drawable_size(
            self.presentation_resolution,
            size.width,
            size.height,
            window.scale_factor(),
        );
        layer.set_drawable_size(CGSize::new(f64::from(width), f64::from(height)));
        eprintln!(
            "[REMu][METAL] presenter started on {} at {width}x{height} ({:?})",
            device.name(),
            self.presentation_resolution
        );

        // A CLI-launched AppKit process can become the active application before its first
        // NSWindow is ordered onto the current Space. Explicitly order and focus the completed
        // window so a running game cannot remain hidden behind the launching application.
        window.set_visible(true);
        window.focus_window();
        self.window = Some(window);
        self.layer = Some(layer);
        Ok(())
    }

    pub(crate) fn update_window_title(&mut self) {
        const UPDATE_INTERVAL: Duration = Duration::from_millis(500);
        let elapsed = self.statistics.interval_started.elapsed();
        if elapsed < UPDATE_INTERVAL {
            return;
        }
        let seconds = elapsed.as_secs_f64();
        let completed = self.statistics.completed_presents;
        let flip_rate = f64::from(self.statistics.guest_flips) / seconds;
        let present_rate = f64::from(completed) / seconds;
        // The user-visible FPS is the rate of new guest frames, not redraws of the most recent
        // drawable. Presentation can keep running while the guest is stalled on rendering, so
        // reporting it as FPS hid the exact black-screen/audio-starvation condition this title is
        // intended to diagnose.
        let frame_milliseconds = if flip_rate > 0.0 {
            1_000.0 / flip_rate
        } else {
            0.0
        };
        let draw_rate = f64::from(self.statistics.encoded_draws) / seconds;
        let compute_rate = f64::from(self.statistics.encoded_compute_dispatches) / seconds;
        let render_pass_rate = f64::from(self.statistics.encoded_render_passes) / seconds;
        let presentation_latency_milliseconds = if completed == 0 {
            0.0
        } else {
            self.statistics.presentation_queue_latency.as_secs_f64() * 1_000.0
                / f64::from(completed)
        };
        let gpu_utilization =
            (self.statistics.gpu_execution_time.as_secs_f64() * 100.0 / seconds).clamp(0.0, 100.0);
        let guest_batch_rate = f64::from(self.statistics.submitted_guest_batches) / seconds;
        let pipeline_barrier_rate =
            f64::from(self.statistics.coalesced_pipeline_barriers) / seconds;
        let process_memory = process_memory_statistics();
        let audio_output = remu_gpu::audio_output_frames();
        let interval_audio_output =
            audio_output.saturating_sub(self.statistics.last_audio_output_frames);
        let audio_output_rate = interval_audio_output as f64 / seconds;
        let audio_underruns = remu_gpu::audio_underrun_frames();
        let interval_underruns =
            audio_underruns.saturating_sub(self.statistics.last_audio_underrun_frames);
        let audio_underrun_percent = if interval_audio_output == 0 {
            0.0
        } else {
            let output = u32::try_from(interval_audio_output).unwrap_or(u32::MAX);
            let underruns = u32::try_from(interval_underruns).unwrap_or(u32::MAX);
            f64::from(underruns) * 100.0 / f64::from(output)
        };
        let memory = process_memory.map_or_else(
            || "RSS ? Foot ?".to_owned(),
            |memory| {
                format!(
                    "RSS {} MiB Foot {}/{} MiB",
                    memory.resident_bytes.div_ceil(1024 * 1024),
                    memory.physical_footprint_bytes.div_ceil(1024 * 1024),
                    memory.peak_physical_footprint_bytes.div_ceil(1024 * 1024),
                )
            },
        );
        let sampled_texture_cache_resident_bytes = self
            .sampled_texture_cache
            .values()
            .fold(0_u64, |total, cached| {
                total.saturating_add(cached.resident_bytes)
            });
        let sampled_texture_cache_guest_bytes = self
            .sampled_texture_cache
            .values()
            .fold(0_u64, |total, cached| {
                total.saturating_add(cached.write_watch.range().1)
            });
        let sampled_texture_cache_mib = sampled_texture_cache_resident_bytes.div_ceil(1024 * 1024);
        let sampled_texture_cache_guest_mib =
            sampled_texture_cache_guest_bytes.div_ceil(1024 * 1024);
        let (metal_allocated_mib, metal_recommended_mib) =
            self.device.as_ref().map_or((0, 0), |device| {
                (
                    u64::try_from(device.current_allocated_size())
                        .unwrap_or(u64::MAX)
                        .div_ceil(1024 * 1024),
                    device
                        .recommended_max_working_set_size()
                        .div_ceil(1024 * 1024),
                )
            });
        let pending_gpu_events = remu_gpu::gpu_work_items_pending();
        let pending_flips = remu_gpu::gpu_flips_pending();
        let title = format!(
            "REMu | FPS {flip_rate:.1} ({frame_milliseconds:.1} ms) | Present {present_rate:.1}/s | GPU {gpu_utilization:.0}% Lat {presentation_latency_milliseconds:.1} ms | {memory} MTL {metal_allocated_mib}/{metal_recommended_mib} MiB | Draw {draw_rate:.0}/s RP {render_pass_rate:.0}/s CS {compute_rate:.0}/s | CB {guest_batch_rate:.0}/s PB {pipeline_barrier_rate:.0}/s ({}) | Q W{} P{} F{} | Tex {}/{sampled_texture_cache_mib} MiB | Pipe {}/{} | Audio {audio_output_rate:.0}/s U {audio_underrun_percent:.1}%",
            self.committed_guest_batches.len(),
            pending_gpu_events,
            self.pending_presentations.len(),
            pending_flips,
            self.sampled_texture_cache.len(),
            self.compiled_shader_pairs.len(),
            self.compiling_shader_pairs.len(),
        );
        if let Some(window) = self.window.as_ref() {
            window.set_title(&title);
        }
        if self.statistics.last_logged.elapsed() >= Duration::from_secs(5) {
            eprintln!("[REMu][PERF] {title}");
            eprintln!(
                "[REMu][PERF-CAUSES] RP-end compute={:.0}/s key={:.0}/s color-op={:.0}/s flush={:.0}/s | CB tick={:.0}/s memory={:.0}/s wait={:.0}/s flip={:.0}/s capacity={:.0}/s pfp={:.0}/s sync={:.0}/s",
                f64::from(self.statistics.render_pass_ends_for_compute) / seconds,
                f64::from(self.statistics.render_pass_ends_for_target_change) / seconds,
                f64::from(self.statistics.render_pass_ends_for_color_operation) / seconds,
                f64::from(self.statistics.render_pass_ends_for_batch_flush) / seconds,
                f64::from(self.statistics.batch_flushes_from_event_loop) / seconds,
                f64::from(self.statistics.batch_flushes_for_guest_memory_write) / seconds,
                f64::from(self.statistics.batch_flushes_for_blocked_wait) / seconds,
                f64::from(self.statistics.batch_flushes_for_flip_snapshot) / seconds,
                f64::from(self.statistics.batch_flushes_for_work_capacity) / seconds,
                f64::from(self.statistics.batch_flushes_for_command_processor_barrier) / seconds,
                f64::from(self.statistics.batch_flushes_for_cpu_synchronization) / seconds,
            );
            let transitions = &self.statistics.render_pass_key_transitions;
            eprintln!(
                "[REMu][PERF-RP-KEY] target count={:.0}/s addr={:.0}/s width={:.0}/s height={:.0}/s pitch={:.0}/s tile={:.0}/s format={:.0}/s bpp={:.0}/s samples={:.0}/s scanout={:.0}/s type={:.0}/s layers={:.0}/s base-layer={:.0}/s mip={:.0}/s mip-count={:.0}/s | depth presence={:.0}/s addr={:.0}/s width={:.0}/s height={:.0}/s pitch={:.0}/s tile={:.0}/s format={:.0}/s bpp={:.0}/s samples={:.0}/s layers={:.0}/s clear={:.0}/s clear-seq={:.0}/s",
                f64::from(transitions.target_count) / seconds,
                f64::from(transitions.target_address) / seconds,
                f64::from(transitions.target_width) / seconds,
                f64::from(transitions.target_height) / seconds,
                f64::from(transitions.target_pitch) / seconds,
                f64::from(transitions.target_tile_mode) / seconds,
                f64::from(transitions.target_pixel_format) / seconds,
                f64::from(transitions.target_bytes_per_pixel) / seconds,
                f64::from(transitions.target_sample_count) / seconds,
                f64::from(transitions.target_direct_scanout) / seconds,
                f64::from(transitions.target_texture_type) / seconds,
                f64::from(transitions.target_layer_count) / seconds,
                f64::from(transitions.target_base_layer) / seconds,
                f64::from(transitions.target_mip_level) / seconds,
                f64::from(transitions.target_mip_level_count) / seconds,
                f64::from(transitions.depth_presence) / seconds,
                f64::from(transitions.depth_address) / seconds,
                f64::from(transitions.depth_width) / seconds,
                f64::from(transitions.depth_height) / seconds,
                f64::from(transitions.depth_pitch) / seconds,
                f64::from(transitions.depth_tile_mode) / seconds,
                f64::from(transitions.depth_pixel_format) / seconds,
                f64::from(transitions.depth_bytes_per_pixel) / seconds,
                f64::from(transitions.depth_sample_count) / seconds,
                f64::from(transitions.depth_layer_count) / seconds,
                f64::from(transitions.depth_clear_value) / seconds,
                f64::from(transitions.depth_clear_sequence) / seconds,
            );
            let compile_enqueued = PIPELINE_COMPILE_ENQUEUED.swap(0, Ordering::Relaxed);
            let compile_queue_full = PIPELINE_COMPILE_QUEUE_FULL.swap(0, Ordering::Relaxed);
            let compile_completed = PIPELINE_COMPILE_COMPLETED.swap(0, Ordering::Relaxed);
            let compile_failed = PIPELINE_COMPILE_FAILED.swap(0, Ordering::Relaxed);
            let draw_gates = PIPELINE_DRAW_GATES.swap(0, Ordering::Relaxed);
            let compute_gates = PIPELINE_COMPUTE_GATES.swap(0, Ordering::Relaxed);
            let (pipeline_blocks, memory_blocks) = self.receiver.block_reason_counts();
            eprintln!(
                "[REMu][PERF-PIPELINE] ready={}/{} active={}/{} failed={}/{} interval=enqueued:{compile_enqueued}/full:{compile_queue_full}/done:{compile_completed}/failed:{compile_failed}/gates:{draw_gates}/{compute_gates} blocked=pipeline:{pipeline_blocks}/memory:{memory_blocks}",
                self.compiled_shader_pairs.len(),
                self.compiled_compute_pipelines.len(),
                self.compiling_shader_pairs.len(),
                self.compiling_compute_pipelines.len(),
                self.failed_shader_pairs.len(),
                self.failed_compute_pipelines.len(),
            );
            let queue_diagnostics = self.receiver.queue_diagnostics();
            if !queue_diagnostics.is_empty() {
                eprintln!("[REMu][PERF-QUEUES] {queue_diagnostics}");
            }
            let wait_dependencies = self.receiver.blocked_wait_dependency_diagnostics();
            if !wait_dependencies.is_empty() {
                eprintln!("[REMu][PERF-QUEUE-WAITS] {wait_dependencies}");
            }
            if !self.committed_guest_batches.is_empty() {
                let batches = self
                    .committed_guest_batches
                    .iter()
                    .enumerate()
                    .map(|(index, batch)| {
                        format!(
                            "{index}:{:?}:[{}]",
                            batch.command_buffer.status(),
                            describe_guest_batch(&batch.work),
                        )
                    })
                    .collect::<Vec<_>>()
                    .join("; ");
                eprintln!("[REMu][PERF-BATCHES] {batches}");
                if trace_metal_command_stalls() {
                    for (index, batch) in self.committed_guest_batches.iter().enumerate() {
                        let Some(age) = batch.committed_at.map(|committed| committed.elapsed())
                        else {
                            continue;
                        };
                        let status = batch.command_buffer.status();
                        if !command_buffer_is_stall_candidate(status, age) {
                            continue;
                        }
                        eprintln!(
                            "[REMu][METAL][STALL] batch={index} status={status:?} age_ms={} {} work=[{}]",
                            age.as_millis(),
                            command_buffer_native_state_summary(&batch.command_buffer),
                            describe_guest_batch(&batch.work),
                        );
                    }
                }
            }
            if !self.dropped_draw_executions.is_empty()
                || !self.dropped_compute_executions.is_empty()
            {
                // Executions that failed after pipeline compilation are dropped; keep the running
                // totals visible in every run's perf trail so a scene pass failing each frame
                // cannot disappear behind its single once-per-key log line.
                let dropped_draws: u64 = self.dropped_draw_executions.values().sum();
                let dropped_dispatches: u64 = self.dropped_compute_executions.values().sum();
                let worst_draw = self
                    .dropped_draw_executions
                    .iter()
                    .max_by_key(|(_, count)| **count)
                    .map_or_else(String::new, |((es, ps), count)| {
                        format!(" worst_draw=es:{es:#x}/ps:{ps:#x}/{count}")
                    });
                let worst_dispatch = self
                    .dropped_compute_executions
                    .iter()
                    .max_by_key(|(_, count)| **count)
                    .map_or_else(String::new, |(cs, count)| {
                        format!(" worst_cs={cs:#x}/{count}")
                    });
                eprintln!(
                    "[REMu][PERF-DROPPED] draws={dropped_draws}/{} keys compute={dropped_dispatches}/{} keys{worst_draw}{worst_dispatch}",
                    self.dropped_draw_executions.len(),
                    self.dropped_compute_executions.len(),
                );
            }
            {
                // Draws consumed without rasterizing report success, so neither the drop counters
                // nor any failure log sees them; these cumulative totals are their only footprint.
                use super::super::stats::{
                    NOOP_DRAWS_COLOR_OPERATION, NOOP_DRAWS_COVER_PLACEHOLDER,
                    NOOP_DRAWS_EMPTY_SCISSOR, NOOP_DRAWS_FAST_CLEAR_NOT_PENDING,
                    NOOP_DRAWS_MESH_DISPATCH_SKIPPED, NOOP_DRAWS_RASTER_DISABLED,
                };
                let noop_draws = [
                    ("cover", &NOOP_DRAWS_COVER_PLACEHOLDER),
                    ("raster-off", &NOOP_DRAWS_RASTER_DISABLED),
                    ("color-op", &NOOP_DRAWS_COLOR_OPERATION),
                    ("empty-scissor", &NOOP_DRAWS_EMPTY_SCISSOR),
                    ("mesh-skip", &NOOP_DRAWS_MESH_DISPATCH_SKIPPED),
                    ("fce-idle", &NOOP_DRAWS_FAST_CLEAR_NOT_PENDING),
                ];
                let summary = noop_draws
                    .iter()
                    .filter_map(|(kind, count)| {
                        let count = count.load(Ordering::Relaxed);
                        (count != 0).then(|| format!("{kind}={count}"))
                    })
                    .collect::<Vec<_>>()
                    .join(" ");
                if !summary.is_empty() {
                    eprintln!("[REMu][PERF-NOOP] draws consumed without rasterizing: {summary}");
                }
            }
            let snapshot_count = TEXTURE_SNAPSHOT_COUNT.swap(0, Ordering::Relaxed);
            let snapshot_bytes = TEXTURE_SNAPSHOT_BYTES.swap(0, Ordering::Relaxed);
            let snapshot_nanoseconds = TEXTURE_SNAPSHOT_NANOSECONDS.swap(0, Ordering::Relaxed);
            let snapshot_unstable = TEXTURE_SNAPSHOT_UNSTABLE_COUNT.swap(0, Ordering::Relaxed);
            let dirty_validation_count = TEXTURE_DIRTY_VALIDATION_COUNT.swap(0, Ordering::Relaxed);
            let dirty_validation_bytes = TEXTURE_DIRTY_VALIDATION_BYTES.swap(0, Ordering::Relaxed);
            let dirty_validation_nanoseconds =
                TEXTURE_DIRTY_VALIDATION_NANOSECONDS.swap(0, Ordering::Relaxed);
            let dirty_validation_unchanged =
                TEXTURE_DIRTY_VALIDATION_UNCHANGED.swap(0, Ordering::Relaxed);
            let decode_count = MODE27_DECODE_COUNT.swap(0, Ordering::Relaxed);
            let decode_bytes = MODE27_DECODE_BYTES.swap(0, Ordering::Relaxed);
            let decode_nanoseconds = MODE27_DECODE_NANOSECONDS.swap(0, Ordering::Relaxed);
            let upload_count = MIP_UPLOAD_COUNT.swap(0, Ordering::Relaxed);
            let upload_bytes = MIP_UPLOAD_BYTES.swap(0, Ordering::Relaxed);
            let upload_nanoseconds = MIP_UPLOAD_NANOSECONDS.swap(0, Ordering::Relaxed);
            eprintln!(
                "[REMu][PERF-TEXTURE] snapshot={snapshot_count}/{:.1}MiB/{:.1}ms/{snapshot_unstable}unstable dirty_validate={dirty_validation_count}/{:.1}MiB/{:.1}ms/{dirty_validation_unchanged}clean mode27={decode_count}/{:.1}MiB/{:.1}ms upload={upload_count}/{:.1}MiB/{:.1}ms",
                snapshot_bytes as f64 / (1024.0 * 1024.0),
                snapshot_nanoseconds as f64 / 1_000_000.0,
                dirty_validation_bytes as f64 / (1024.0 * 1024.0),
                dirty_validation_nanoseconds as f64 / 1_000_000.0,
                decode_bytes as f64 / (1024.0 * 1024.0),
                decode_nanoseconds as f64 / 1_000_000.0,
                upload_bytes as f64 / (1024.0 * 1024.0),
                upload_nanoseconds as f64 / 1_000_000.0,
            );
            let cache_hits = SAMPLED_TEXTURE_CACHE_HITS.swap(0, Ordering::Relaxed);
            let cache_misses = SAMPLED_TEXTURE_CACHE_MISSES.swap(0, Ordering::Relaxed);
            let cache_dirty_reloads =
                SAMPLED_TEXTURE_CACHE_DIRTY_RELOADS.swap(0, Ordering::Relaxed);
            let cache_acquire_invalidations =
                SAMPLED_TEXTURE_CACHE_ACQUIRE_INVALIDATIONS.swap(0, Ordering::Relaxed);
            let cache_capacity_evictions =
                SAMPLED_TEXTURE_CACHE_CAPACITY_EVICTIONS.swap(0, Ordering::Relaxed);
            let resident_acquire_invalidations =
                RESIDENT_TEXTURE_ACQUIRE_INVALIDATIONS.swap(0, Ordering::Relaxed);
            let depth_resident_hits = RESIDENT_DEPTH_SAMPLE_HITS.swap(0, Ordering::Relaxed);
            let alias_writeback_count = SAMPLED_ALIAS_WRITEBACK_COUNT.swap(0, Ordering::Relaxed);
            let alias_writeback_bytes = SAMPLED_ALIAS_WRITEBACK_BYTES.swap(0, Ordering::Relaxed);
            let alias_writeback_nanoseconds =
                SAMPLED_ALIAS_WRITEBACK_NANOSECONDS.swap(0, Ordering::Relaxed);
            let quarantine_adopted = TEXTURE_QUARANTINE_ADOPTED.swap(0, Ordering::Relaxed);
            let quarantine_rejected = TEXTURE_QUARANTINE_REJECTED.swap(0, Ordering::Relaxed);
            // Expected to stay zero: any count is a GPU write that reached no resident at all.
            let write_attribution_misses =
                RESIDENT_WRITE_ATTRIBUTION_MISSES.swap(0, Ordering::Relaxed);
            eprintln!(
                "[REMu][PERF-TEXTURE-CACHE] resident={}/{}MiB guest={}MiB metal={}/{}MiB watches={} hit={} miss={} dirty_reload={} acquire_invalidate={} resident_acquire_invalidate={} capacity_evict={} depth_native={} alias_writeback={}/{:.1}MiB/{:.1}ms quarantine={}/{} adopt={} stale={} write_attribution_miss={}",
                self.sampled_texture_cache.len(),
                sampled_texture_cache_mib,
                sampled_texture_cache_guest_mib,
                metal_allocated_mib,
                metal_recommended_mib,
                self.sampled_texture_write_watches.len(),
                cache_hits,
                cache_misses,
                cache_dirty_reloads,
                cache_acquire_invalidations,
                resident_acquire_invalidations,
                cache_capacity_evictions,
                depth_resident_hits,
                alias_writeback_count,
                alias_writeback_bytes as f64 / (1024.0 * 1024.0),
                alias_writeback_nanoseconds as f64 / 1_000_000.0,
                self.retired_texture_quarantine.len(),
                RETIRED_TEXTURE_QUARANTINE_CAPACITY,
                quarantine_adopted,
                quarantine_rejected,
                write_attribution_misses,
            );
            // Host→guest publication of rendered pixels, by what asked for it. `drop` is the
            // traffic REmu refused to stage into a range no guest read named; `pub` is the
            // remaining exposure, and every one of those destinations was named by the guest.
            let writeback_published_bytes =
                RESIDENT_WRITEBACK_PUBLISHED_BYTES.swap(0, Ordering::Relaxed);
            let writeback_discarded_bytes =
                RESIDENT_WRITEBACK_DISCARDED_BYTES.swap(0, Ordering::Relaxed);
            let writeback_detail = WRITEBACK_TRIGGERS
                .iter()
                .map(|trigger| {
                    format!(
                        " {}={}/{}",
                        trigger.label(),
                        RESIDENT_WRITEBACK_PUBLISHED[trigger.index()].swap(0, Ordering::Relaxed),
                        RESIDENT_WRITEBACK_DISCARDED[trigger.index()].swap(0, Ordering::Relaxed),
                    )
                })
                .collect::<String>();
            eprintln!(
                "[REMu][PERF-TEXTURE-WRITEBACK] published={:.1}MiB dropped={:.1}MiB pub/drop:{writeback_detail}",
                writeback_published_bytes as f64 / (1024.0 * 1024.0),
                writeback_discarded_bytes as f64 / (1024.0 * 1024.0),
            );
            // The range algebra's own accounting. `owned` is the authority the residents hold and
            // `debt` the part of it that exists only in Metal; the counters are the ways a claim
            // stopped being exact. The demotion, re-materialisation and publish figures arrive
            // with the paths that produce them rather than as standing zeros.
            let (mut live, mut stale, mut stale_bytes) = (0_usize, 0_usize, 0_u64);
            let (mut owned_bytes, mut debt_bytes, mut owned_runs) = (0_u64, 0_u64, 0_usize);
            for resident in self.resident_guest_textures.values() {
                match resident.state {
                    ResidencyState::Live => live += 1,
                    ResidencyState::Stale(_) => {
                        stale += 1;
                        stale_bytes = stale_bytes.saturating_add(
                            u64::try_from(resident.texture.allocated_size()).unwrap_or(u64::MAX),
                        );
                    }
                }
                owned_bytes = owned_bytes.saturating_add(resident.owned.byte_count());
                debt_bytes = debt_bytes.saturating_add(resident.gpu_authored.byte_count());
                owned_runs += resident.owned.runs().len();
            }
            eprintln!(
                "[REMu][PERF-TEXTURE-OWNED] live={live} stale={stale} owned={:.1}MiB/{owned_runs} runs debt={:.1}MiB survived={} demoted={} reclaim_discard={} coarsened={} unresolved={} fallback={}",
                owned_bytes as f64 / (1024.0 * 1024.0),
                debt_bytes as f64 / (1024.0 * 1024.0),
                OWNED_SUBTRACT_TENANTS_SURVIVED.swap(0, Ordering::Relaxed),
                OWNED_DEMOTIONS.swap(0, Ordering::Relaxed),
                OWNED_RECLAIM_DISCARD.swap(0, Ordering::Relaxed),
                OWNED_RANGE_SET_COARSENED.swap(0, Ordering::Relaxed),
                OWNED_UNRESOLVED_SPAN.swap(0, Ordering::Relaxed),
                OWNED_FALLBACK_FULL_RANGE.swap(0, Ordering::Relaxed),
            );
            // Demoted allocations pin their Metal storage until the pressure GC's first tier
            // reaches them, so an unbounded population is a leak, not a cache.
            if stale > 128 {
                eprintln!(
                    "[REMu][METAL] {stale} demoted resident textures are pinning {:.1} MiB; the resident budget is not reclaiming them",
                    stale_bytes as f64 / (1024.0 * 1024.0),
                );
            }
            let texture_allocated_bytes = |texture: &metal::Texture| {
                u64::try_from(texture.allocated_size()).unwrap_or(u64::MAX)
            };
            let resident_target_bytes =
                self.resident_guest_textures
                    .values()
                    .fold(0_u64, |total, resident| {
                        // A component-reorder texture is a view of this allocation, so count only
                        // the owning resident texture.
                        total.saturating_add(texture_allocated_bytes(&resident.texture))
                    });
            let target_ownership = resident_texture_ownership_stats(&self.resident_guest_textures);
            let depth_array_bytes = self
                .depth_array_textures
                .values()
                .fold(0_u64, |total, resident| {
                    total.saturating_add(texture_allocated_bytes(&resident.texture))
                });
            let direct_scanout_bytes = self
                .direct_scanout_textures
                .values()
                .fold(0_u64, |total, resident| {
                    total.saturating_add(texture_allocated_bytes(&resident.texture))
                });
            let scanout_snapshot_bytes = self
                .scanout_snapshot_textures
                .values()
                .fold(0_u64, |total, resident| {
                    total.saturating_add(texture_allocated_bytes(&resident.texture))
                });
            let multisample_bytes = self
                .multisample_textures
                .values()
                .fold(0_u64, |total, resident| {
                    total.saturating_add(texture_allocated_bytes(&resident.texture))
                });
            let fallback_bytes = self
                .fallback_texture_cache
                .values()
                .fold(0_u64, |total, texture| {
                    total.saturating_add(texture_allocated_bytes(texture))
                });
            let accounted_texture_bytes = sampled_texture_cache_resident_bytes
                .saturating_add(resident_target_bytes)
                .saturating_add(depth_array_bytes)
                .saturating_add(direct_scanout_bytes)
                .saturating_add(scanout_snapshot_bytes)
                .saturating_add(multisample_bytes)
                .saturating_add(fallback_bytes);
            let metal_allocated_bytes = metal_allocated_mib.saturating_mul(1024 * 1024);
            let unaccounted_metal_bytes =
                metal_allocated_bytes.saturating_sub(accounted_texture_bytes);
            eprintln!(
                "[REMu][PERF-METAL-RESIDENCY] sampled={}/{:.1}MiB targets={}/{:.1}MiB depth_arrays={}/{:.1}MiB direct_scanout={}/{:.1}MiB scanout_snapshots={}/{:.1}MiB msaa={}/{:.1}MiB fallback={}/{:.1}MiB accounted={:.1}MiB other={:.1}MiB",
                self.sampled_texture_cache.len(),
                sampled_texture_cache_resident_bytes as f64 / (1024.0 * 1024.0),
                self.resident_guest_textures.len(),
                resident_target_bytes as f64 / (1024.0 * 1024.0),
                self.depth_array_textures.len(),
                depth_array_bytes as f64 / (1024.0 * 1024.0),
                self.direct_scanout_textures.len(),
                direct_scanout_bytes as f64 / (1024.0 * 1024.0),
                self.scanout_snapshot_textures.len(),
                scanout_snapshot_bytes as f64 / (1024.0 * 1024.0),
                self.multisample_textures.len(),
                multisample_bytes as f64 / (1024.0 * 1024.0),
                self.fallback_texture_cache.len(),
                fallback_bytes as f64 / (1024.0 * 1024.0),
                accounted_texture_bytes as f64 / (1024.0 * 1024.0),
                unaccounted_metal_bytes as f64 / (1024.0 * 1024.0),
            );
            let mut largest_targets = self
                .resident_guest_textures
                .values()
                .map(|resident| {
                    (
                        texture_allocated_bytes(&resident.texture),
                        resident.descriptor,
                        resident.gpu_written(),
                        resident.content_version,
                        resident.write_watch.retirement(),
                    )
                })
                .collect::<Vec<_>>();
            largest_targets.sort_unstable_by_key(|(bytes, ..)| std::cmp::Reverse(*bytes));
            let largest_targets = largest_targets
                .into_iter()
                .take(8)
                .map(|(bytes, descriptor, gpu_written, version, retirement)| {
                    format!(
                        "{:#x}/{}x{}/p{}/{:?}/tile{}={:.1}MiB/gpu{gpu_written}/v{version}/{retirement:?}",
                        descriptor.address,
                        descriptor.width,
                        descriptor.height,
                        descriptor.pitch,
                        descriptor.pixel_format,
                        descriptor.tile_mode,
                        bytes as f64 / (1024.0 * 1024.0),
                    )
                })
                .collect::<Vec<_>>()
                .join("; ");
            eprintln!(
                "[REMu][PERF-METAL-TARGETS] owner=gpu:{}/{:.1}MiB,cpu:{}/{:.1}MiB mapping=active:{}/{:.1}MiB,reprotected:{}/{:.1}MiB,reprotecting:{}/{:.1}MiB,replaced:{}/{:.1}MiB backing=bases:{}/+{}versions,spans:{}/+{}duplicates,unresolved:{}/{:.1}MiB top=[{}]",
                target_ownership.gpu_owned_count,
                target_ownership.gpu_owned_bytes as f64 / (1024.0 * 1024.0),
                target_ownership.cpu_owned_count,
                target_ownership.cpu_owned_bytes as f64 / (1024.0 * 1024.0),
                target_ownership.active_mapping_count,
                target_ownership.active_mapping_bytes as f64 / (1024.0 * 1024.0),
                target_ownership.reprotected_mapping_count,
                target_ownership.reprotected_mapping_bytes as f64 / (1024.0 * 1024.0),
                target_ownership.reprotecting_mapping_count,
                target_ownership.reprotecting_mapping_bytes as f64 / (1024.0 * 1024.0),
                target_ownership.replaced_mapping_count,
                target_ownership.replaced_mapping_bytes as f64 / (1024.0 * 1024.0),
                target_ownership.unique_backing_bases,
                target_ownership.duplicate_backing_bases,
                target_ownership.unique_backing_spans,
                target_ownership.duplicate_backing_spans,
                target_ownership.unresolved_backing_count,
                target_ownership.unresolved_backing_bytes as f64 / (1024.0 * 1024.0),
                largest_targets,
            );
            if resident_texture_lifetime_diagnostics_enabled() {
                let current_work = resident_texture_activity_latest_work();
                let current_flip = self.next_flip_sequence;
                let activity = resident_texture_activity_stats(
                    &self.resident_guest_textures,
                    current_work,
                    current_flip,
                );
                macro_rules! format_age_buckets {
                    ($labels:expr, $buckets:expr) => {
                        $labels
                            .iter()
                            .zip($buckets.iter())
                            .map(|(label, bucket)| {
                                format!(
                                    "{label}:{}/{:.1}MiB(gpu:{}/{:.1},cpu:{}/{:.1})",
                                    bucket.resident_count,
                                    bucket.resident_bytes as f64 / (1024.0 * 1024.0),
                                    bucket.gpu_owned_count,
                                    bucket.gpu_owned_bytes as f64 / (1024.0 * 1024.0),
                                    bucket.cpu_owned_count,
                                    bucket.cpu_owned_bytes as f64 / (1024.0 * 1024.0),
                                )
                            })
                            .collect::<Vec<_>>()
                            .join(",")
                    };
                }
                eprintln!(
                    "[REMu][PERF-METAL-TARGET-ACTIVITY] now=work:{current_work}/flip:{current_flip} read_work=[{}] write_work=[{}] read_flip=[{}] write_flip=[{}]",
                    format_age_buckets!(RESIDENT_WORK_AGE_LABELS, activity.read_work),
                    format_age_buckets!(RESIDENT_WORK_AGE_LABELS, activity.write_work),
                    format_age_buckets!(RESIDENT_FLIP_AGE_LABELS, activity.read_flip),
                    format_age_buckets!(RESIDENT_FLIP_AGE_LABELS, activity.write_flip),
                );

                let overlap = resident_texture_backing_overlap_stats(&self.resident_guest_textures);
                eprintln!(
                    "[REMu][PERF-METAL-TARGET-OVERLAP] pairs={}/{:.1}MiB residents={}/{:.1}MiB gpu={}/{:.1}MiB cpu={}/{:.1}MiB",
                    overlap.pair_count,
                    overlap.pair_overlap_bytes as f64 / (1024.0 * 1024.0),
                    overlap.resident_count,
                    overlap.resident_bytes as f64 / (1024.0 * 1024.0),
                    overlap.gpu_owned_count,
                    overlap.gpu_owned_bytes as f64 / (1024.0 * 1024.0),
                    overlap.cpu_owned_count,
                    overlap.cpu_owned_bytes as f64 / (1024.0 * 1024.0),
                );

                let mut oldest = self
                    .resident_guest_textures
                    .values()
                    .filter_map(|resident| {
                        let activity = resident.activity?;
                        let last_work = activity.last_read_work.max(activity.last_write_work);
                        let last_flip = activity.last_read_flip.max(activity.last_write_flip);
                        Some((last_work, last_flip, resident, activity))
                    })
                    .collect::<Vec<_>>();
                oldest.sort_unstable_by_key(|(last_work, last_flip, ..)| {
                    (
                        last_work.is_some(),
                        *last_work,
                        last_flip.is_some(),
                        *last_flip,
                    )
                });
                let display_stamp = |work: Option<u64>, flip: Option<u64>| {
                    work.zip(flip).map_or_else(
                        || "never".to_owned(),
                        |(work, flip)| format!("{work}@f{flip}"),
                    )
                };
                let oldest = oldest
                    .into_iter()
                    .take(16)
                    .map(|(last_work, last_flip, resident, activity)| {
                        format!(
                            "{:#x}/{}x{}/{:?}/tile{}={:.1}MiB/gpu{}/v{}/r{}/w{}/age{}w+{}f",
                            resident.descriptor.address,
                            resident.descriptor.width,
                            resident.descriptor.height,
                            resident.descriptor.pixel_format,
                            resident.descriptor.tile_mode,
                            texture_allocated_bytes(&resident.texture) as f64 / (1024.0 * 1024.0),
                            resident.gpu_written(),
                            resident.content_version,
                            display_stamp(activity.last_read_work, activity.last_read_flip),
                            display_stamp(activity.last_write_work, activity.last_write_flip),
                            last_work.map_or_else(
                                || "never".to_owned(),
                                |last| current_work.saturating_sub(last).to_string(),
                            ),
                            last_flip.map_or_else(
                                || "never".to_owned(),
                                |last| current_flip.saturating_sub(last).to_string(),
                            ),
                        )
                    })
                    .collect::<Vec<_>>()
                    .join("; ");
                eprintln!("[REMu][PERF-METAL-TARGET-IDLE] oldest=[{oldest}]");
            }
            eprintln!(
                "[REMu][PERF-GLOBAL-CACHE] captured={}/{:.1}MiB identity={}/{:.1}MiB",
                self.global_buffer_cache.len(),
                self.global_buffer_cache_bytes as f64 / (1024.0 * 1024.0),
                self.guest_global_buffer_cache.len(),
                self.guest_global_buffer_cache_bytes as f64 / (1024.0 * 1024.0),
            );
            let snapshot_details = take_texture_snapshot_work_details();
            if !snapshot_details.is_empty() {
                let top = snapshot_details
                    .into_iter()
                    .take(8)
                    .map(|(key, aggregate)| {
                        format!(
                            "{:#x}/{:.1}MiB@L{}={}/{:.1}MiB/{:.1}ms",
                            key.address,
                            key.byte_count as f64 / (1024.0 * 1024.0),
                            key.caller_line,
                            aggregate.count,
                            aggregate.bytes as f64 / (1024.0 * 1024.0),
                            aggregate.nanoseconds as f64 / 1_000_000.0,
                        )
                    })
                    .collect::<Vec<_>>()
                    .join("; ");
                eprintln!("[REMu][PERF-SNAPSHOT-TOP] {top}");
            }
            let dirty_details = take_texture_dirty_work_details();
            if !dirty_details.is_empty() {
                let top = dirty_details
                    .into_iter()
                    .take(8)
                    .map(|(key, aggregate)| {
                        let dirty_percent = if aggregate.full_bytes == 0 {
                            0.0
                        } else {
                            aggregate.dirty_bytes as f64 * 100.0
                                / aggregate.full_bytes as f64
                        };
                        format!(
                            "{:#x}/{:.1}MiB@L{}={}snap/{}clean/{}imprecise dirty={:.1}MiB({dirty_percent:.3}%)/{}pages",
                            key.address,
                            key.byte_count as f64 / (1024.0 * 1024.0),
                            key.caller_line,
                            aggregate.snapshots,
                            aggregate.clean_snapshots,
                            aggregate.imprecise_snapshots,
                            aggregate.dirty_bytes as f64 / (1024.0 * 1024.0),
                            aggregate.dirty_pages,
                        )
                    })
                    .collect::<Vec<_>>()
                    .join("; ");
                eprintln!("[REMu][PERF-TEXTURE-DIRTY-TOP] {top}");
            }
            let mode27_details = take_mode27_work_details();
            if !mode27_details.is_empty() {
                let top = mode27_details
                    .into_iter()
                    .take(8)
                    .map(|(key, aggregate)| {
                        format!(
                            "{:?}@{:#x}/{}x{}/p{}/fmt{:?}/m{}={}/{:.1}MiB/{:.1}ms",
                            key.origin,
                            key.descriptor.address,
                            key.descriptor.width,
                            key.descriptor.height,
                            key.descriptor.pitch,
                            key.descriptor.pixel_format,
                            key.mip_level_count,
                            aggregate.count,
                            aggregate.bytes as f64 / (1024.0 * 1024.0),
                            aggregate.nanoseconds as f64 / 1_000_000.0,
                        )
                    })
                    .collect::<Vec<_>>()
                    .join("; ");
                eprintln!("[REMu][PERF-TEXTURE-TOP] {top}");
            }
            self.statistics.last_logged = Instant::now();
        }
        self.statistics.interval_started = Instant::now();
        self.statistics.guest_flips = 0;
        self.statistics.completed_presents = 0;
        self.statistics.submitted_guest_batches = 0;
        self.statistics.coalesced_pipeline_barriers = 0;
        self.statistics.encoded_draws = 0;
        self.statistics.encoded_compute_dispatches = 0;
        self.statistics.encoded_render_passes = 0;
        self.statistics.render_pass_ends_for_compute = 0;
        self.statistics.render_pass_ends_for_target_change = 0;
        self.statistics.render_pass_ends_for_color_operation = 0;
        self.statistics.render_pass_ends_for_batch_flush = 0;
        self.statistics.render_pass_key_transitions = Default::default();
        self.statistics.batch_flushes_from_event_loop = 0;
        self.statistics.batch_flushes_for_guest_memory_write = 0;
        self.statistics.batch_flushes_for_blocked_wait = 0;
        self.statistics.batch_flushes_for_flip_snapshot = 0;
        self.statistics.batch_flushes_for_work_capacity = 0;
        self.statistics.batch_flushes_for_command_processor_barrier = 0;
        self.statistics.batch_flushes_for_cpu_synchronization = 0;
        self.statistics.gpu_execution_time = Duration::ZERO;
        self.statistics.presentation_queue_latency = Duration::ZERO;
        self.statistics.last_audio_output_frames = audio_output;
        self.statistics.last_audio_underrun_frames = audio_underruns;
    }

    pub(crate) fn update_keyboard_input(&mut self, code: KeyCode, state: ElementState) {
        match state {
            ElementState::Pressed => {
                self.pressed_keys.insert(code);
            }
            ElementState::Released => {
                self.pressed_keys.remove(&code);
            }
        }
        self.publish_pad_state();
    }

    pub(crate) fn publish_pad_state(&self) {
        let down = |code| self.pressed_keys.contains(&code);
        let mut buttons = 0;
        for (pressed, button) in [
            (down(KeyCode::ArrowUp), PAD_BUTTON_UP),
            (down(KeyCode::ArrowRight), PAD_BUTTON_RIGHT),
            (down(KeyCode::ArrowDown), PAD_BUTTON_DOWN),
            (down(KeyCode::ArrowLeft), PAD_BUTTON_LEFT),
            (
                down(KeyCode::Enter) || down(KeyCode::NumpadEnter) || down(KeyCode::KeyZ),
                PAD_BUTTON_CROSS,
            ),
            (
                down(KeyCode::Escape) || down(KeyCode::KeyX),
                PAD_BUTTON_CIRCLE,
            ),
            (down(KeyCode::KeyC), PAD_BUTTON_SQUARE),
            (down(KeyCode::KeyV), PAD_BUTTON_TRIANGLE),
            (down(KeyCode::KeyQ), PAD_BUTTON_L1),
            (down(KeyCode::KeyE), PAD_BUTTON_R1),
            (down(KeyCode::KeyR), PAD_BUTTON_L2),
            (down(KeyCode::KeyF), PAD_BUTTON_R2),
            (
                down(KeyCode::Tab) || down(KeyCode::Backspace),
                PAD_BUTTON_OPTIONS,
            ),
        ] {
            if pressed {
                buttons |= button;
            }
        }
        let axis = |negative, positive| match (down(negative), down(positive)) {
            (true, false) => 0,
            (false, true) => 255,
            _ => 128,
        };
        set_host_pad_state(HostPadState {
            buttons,
            left_x: axis(KeyCode::KeyA, KeyCode::KeyD),
            left_y: axis(KeyCode::KeyW, KeyCode::KeyS),
            right_x: axis(KeyCode::KeyJ, KeyCode::KeyL),
            right_y: axis(KeyCode::KeyI, KeyCode::KeyK),
            l2: if down(KeyCode::KeyR) { 255 } else { 0 },
            r2: if down(KeyCode::KeyF) { 255 } else { 0 },
        });
    }
}

fn command_buffer_is_stall_candidate(status: MTLCommandBufferStatus, age: Duration) -> bool {
    age >= Duration::from_secs(1)
        && !matches!(
            status,
            MTLCommandBufferStatus::Completed | MTLCommandBufferStatus::Error
        )
}

pub(crate) fn headless_metal_value_is_enabled(value: Option<&OsStr>) -> bool {
    value.is_some_and(|value| value == "1")
}

const APPKIT_PERSISTENCE_IGNORE_STATE_KEY: &std::ffi::CStr = c"ApplePersistenceIgnoreState";

pub(crate) fn disable_appkit_state_restoration() {
    // AppKit consults this per-application default while handling its initial reopen Apple event.
    // Set it before winit creates NSApplication so crash history from an earlier forced shutdown
    // cannot open a modal persistent-state prompt before ApplicationHandler::resumed. The
    // presenter window is also marked non-restorable when it is created, which prevents new
    // restoration archives from being produced.
    objc::rc::autoreleasepool(|| unsafe {
        let defaults: *mut Object = msg_send![class!(NSUserDefaults), standardUserDefaults];
        let key: *mut Object = msg_send![
            class!(NSString),
            stringWithUTF8String: APPKIT_PERSISTENCE_IGNORE_STATE_KEY.as_ptr()
        ];
        let _: () = msg_send![defaults, setBool: YES forKey: key];
    });
}

#[cfg(test)]
pub(crate) fn appkit_persistent_state_is_ignored() -> bool {
    objc::rc::autoreleasepool(|| unsafe {
        let defaults: *mut Object = msg_send![class!(NSUserDefaults), standardUserDefaults];
        let key: *mut Object = msg_send![
            class!(NSString),
            stringWithUTF8String: APPKIT_PERSISTENCE_IGNORE_STATE_KEY.as_ptr()
        ];
        let ignored: objc::runtime::BOOL = msg_send![defaults, boolForKey: key];
        ignored == YES
    })
}

#[cfg(test)]
pub(crate) fn clear_appkit_persistent_state_override() {
    objc::rc::autoreleasepool(|| unsafe {
        let defaults: *mut Object = msg_send![class!(NSUserDefaults), standardUserDefaults];
        let key: *mut Object = msg_send![
            class!(NSString),
            stringWithUTF8String: APPKIT_PERSISTENCE_IGNORE_STATE_KEY.as_ptr()
        ];
        let _: () = msg_send![defaults, removeObjectForKey: key];
    });
}

pub(crate) fn headless_metal_requested() -> bool {
    headless_metal_value_is_enabled(std::env::var_os("REMU_HEADLESS_METAL").as_deref())
}

pub(crate) fn presenter_window_attributes() -> winit::window::WindowAttributes {
    Window::default_attributes()
        .with_title("REmu")
        .with_inner_size(LogicalSize::new(1280, 720))
        .with_active(true)
}

impl ApplicationHandler for MetalPresenter {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.window.is_some() || self.startup_error.is_some() {
            return;
        }
        if let Err(error) = self.initialize(event_loop) {
            eprintln!("[REMu][METAL] {error}");
            self.startup_error = Some(error);
            event_loop.exit();
        }
    }

    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        _window_id: WindowId,
        event: WindowEvent,
    ) {
        match event {
            WindowEvent::CloseRequested => {
                eprintln!("[REMu][METAL] window close requested by AppKit");
                self.persist_render_pipeline_archive();
                event_loop.exit();
            }
            WindowEvent::Focused(false) => {
                self.pressed_keys.clear();
                self.publish_pad_state();
            }
            WindowEvent::KeyboardInput { event, .. } => {
                if let PhysicalKey::Code(code) = event.physical_key {
                    self.update_keyboard_input(code, event.state);
                }
            }
            WindowEvent::Resized(size) => {
                if size.width != 0
                    && size.height != 0
                    && let Some(layer) = self.layer.as_ref()
                {
                    let scale_factor = self.window.as_ref().map_or(1.0, Window::scale_factor);
                    let (width, height) = presentation_drawable_size(
                        self.presentation_resolution,
                        size.width,
                        size.height,
                        scale_factor,
                    );
                    layer.set_drawable_size(CGSize::new(f64::from(width), f64::from(height)));
                    self.redraw_pending = true;
                }
            }
            WindowEvent::ScaleFactorChanged { scale_factor, .. } => {
                if let (Some(window), Some(layer)) = (self.window.as_ref(), self.layer.as_ref()) {
                    let size = window.inner_size();
                    let (width, height) = presentation_drawable_size(
                        self.presentation_resolution,
                        size.width,
                        size.height,
                        scale_factor,
                    );
                    layer.set_drawable_size(CGSize::new(f64::from(width), f64::from(height)));
                    self.redraw_pending = true;
                }
            }
            WindowEvent::RedrawRequested => self.render(),
            _ => {}
        }
    }

    fn about_to_wait(&mut self, event_loop: &ActiveEventLoop) {
        if self.receive_events() == PresenterFlow::Shutdown {
            event_loop.exit();
            return;
        }
        if let Err(error) = self
            .flush_guest_command_batch(FlushPolicy::Opportunistic, GuestBatchFlushReason::EventLoop)
        {
            eprintln!("[REMu][METAL] guest command batch submission failed: {error}");
        }
        self.retire_completed_presentations();
        self.update_window_title();
        crate::presenter::host_ui::poll_host_ui();
        if self.redraw_pending
            && let Some(window) = self.window.as_ref()
        {
            window.request_redraw();
        }
        // Keep draining runnable guest work without depending on AppKit's timer precision.
        // macOS coalesces the nominal one-millisecond WaitUntil to roughly a display tick under
        // load; combined with GPU_EVENTS_PER_TICK, that imposed a hard command-throughput ceiling
        // while the presenter thread spent most of its time asleep. Dependency-blocked and empty
        // queues still take the timed idle path, so a static screen does not busy-spin.
        if self.receiver.has_runnable_events() {
            event_loop.set_control_flow(ControlFlow::Poll);
        } else {
            event_loop
                .set_control_flow(ControlFlow::WaitUntil(Instant::now() + EVENT_POLL_INTERVAL));
        }
    }
}

pub(crate) fn run_headless_presenter(receiver: Receiver<GpuEvent>) -> Result<(), String> {
    eprintln!("[REMu][METAL] starting headless Metal event loop");
    let mut presenter = MetalPresenter::new(receiver)?;
    eprintln!(
        "[REMu][METAL] headless presenter started on {}",
        presenter
            .device
            .as_ref()
            .ok_or("Metal presenter resources are unavailable")?
            .name()
    );
    loop {
        let (flow, submission) = objc::rc::autoreleasepool(|| {
            let flow = presenter.receive_events();
            let submission = if flow == PresenterFlow::Continue {
                presenter.flush_guest_command_batch(
                    FlushPolicy::Opportunistic,
                    GuestBatchFlushReason::EventLoop,
                )
            } else {
                Ok(())
            };
            (flow, submission)
        });
        submission.map_err(|error| format!("guest command batch submission failed: {error}"))?;
        presenter.retire_completed_presentations();
        // `render` performs pending-flip upload and translated-scanout capture before it checks
        // for a CAMetalLayer. This keeps headless traces faithful through the last guest-visible
        // image stage while intentionally omitting only the host window composite.
        presenter.render();
        presenter.update_window_title();
        if flow == PresenterFlow::Shutdown {
            eprintln!("[REMu][METAL] headless event loop stopped");
            return Ok(());
        }
        std::thread::sleep(EVENT_POLL_INTERVAL);
    }
}
