// Copyright (C) 2026 REmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

use core_graphics_types::geometry::CGSize;
use metal::{
    CompileOptions, Device, MTLClearColor, MTLCommandBufferStatus, MTLLoadAction, MTLOrigin,
    MTLPixelFormat, MTLPrimitiveType, MTLRegion, MTLSamplerAddressMode, MTLSamplerMinMagFilter,
    MTLSamplerMipFilter, MTLSize, MTLStorageMode, MTLStoreAction, MTLTextureType, MTLTextureUsage,
    MTLViewport, MetalLayer, RenderPassDescriptor, RenderPipelineDescriptor, SamplerDescriptor,
    Texture, TextureDescriptor,
};
use objc::runtime::{Object, YES};
use objc::{msg_send, sel, sel_impl};
use remu_gpu::{
    GpuEvent, GpuHostImage, HostPadState, PAD_BUTTON_CIRCLE, PAD_BUTTON_CROSS, PAD_BUTTON_DOWN,
    PAD_BUTTON_L1, PAD_BUTTON_L2, PAD_BUTTON_LEFT, PAD_BUTTON_OPTIONS, PAD_BUTTON_R1,
    PAD_BUTTON_R2, PAD_BUTTON_RIGHT, PAD_BUTTON_SQUARE, PAD_BUTTON_TRIANGLE, PAD_BUTTON_UP,
    set_host_pad_state,
};
use std::collections::HashSet;
use std::ffi::{OsStr, c_void};
use std::sync::atomic::Ordering;
use std::sync::mpsc::Receiver;
use std::sync::{Mutex, OnceLock};
use std::time::{Duration, Instant};
use winit::application::ApplicationHandler;
use winit::dpi::LogicalSize;
use winit::event::{ElementState, WindowEvent};
use winit::event_loop::{ActiveEventLoop, ControlFlow};
use winit::keyboard::{KeyCode, PhysicalKey};
use winit::raw_window_handle::{HasWindowHandle, RawWindowHandle};
use winit::window::{Window, WindowId};

use super::{
    ActiveTranslatedScanout, EVENT_POLL_INTERVAL, GuestBatchFlushReason, GuestTextureDescriptor,
    HostSplashTexture, MIP_UPLOAD_BYTES, MIP_UPLOAD_COUNT, MIP_UPLOAD_NANOSECONDS,
    MODE27_DECODE_BYTES, MODE27_DECODE_COUNT, MODE27_DECODE_NANOSECONDS, MetalPresenter,
    OrderedEventSource, PIPELINE_COMPILE_COMPLETED, PIPELINE_COMPILE_ENQUEUED,
    PIPELINE_COMPILE_FAILED, PIPELINE_COMPILE_QUEUE_FULL, PIPELINE_DRAW_GATES, PendingPresentation,
    PresentationResolution, PresentationResources, PresenterFlow, RESIDENT_DEPTH_SAMPLE_HITS,
    RESIDENT_TEXTURE_ACQUIRE_INVALIDATIONS, SAMPLED_ALIAS_WRITEBACK_BYTES,
    SAMPLED_ALIAS_WRITEBACK_COUNT, SAMPLED_ALIAS_WRITEBACK_NANOSECONDS,
    SAMPLED_TEXTURE_CACHE_ACQUIRE_INVALIDATIONS, SAMPLED_TEXTURE_CACHE_CAPACITY_EVICTIONS,
    SAMPLED_TEXTURE_CACHE_DIRTY_RELOADS, SAMPLED_TEXTURE_CACHE_HITS, SAMPLED_TEXTURE_CACHE_MISSES,
    TEXTURE_SNAPSHOT_BYTES, TEXTURE_SNAPSHOT_COUNT, TEXTURE_SNAPSHOT_NANOSECONDS,
    WriteWatchRefresh, capture_interval, compile_metal_library_with_tools,
    completed_command_buffer_gpu_duration, decode_guest_texture_linear_from_raw,
    diagnostic_scanout_writer, diagnostic_texture_writer, process_resident_bytes,
    raw_video_out_texture_descriptor, read_texture_linear, refresh_resident_texture_write_watch,
    retire_multisample_textures_overlapping_writeback,
    retire_resident_textures_overlapping_writeback, retire_sampled_textures_overlapping_writeback,
    snapshot_guest_texture_raw, take_mode27_work_details, take_texture_dirty_work_details,
    take_texture_snapshot_work_details, trace_scanout_writes,
    write_resident_texture_contents_to_guest,
};

impl MetalPresenter {
    pub(super) fn initialize(&mut self, event_loop: &ActiveEventLoop) -> Result<(), String> {
        let window = event_loop
            .create_window(presenter_window_attributes())
            .map_err(|error| format!("could not create the macOS window: {error}"))?;
        self.initialize_window(window)
    }

    pub(super) fn initialize_window(&mut self, window: Window) -> Result<(), String> {
        let device = self
            .device
            .as_ref()
            .ok_or("Metal presenter resources are unavailable")?;
        let mut layer = MetalLayer::new();
        layer.set_device(device);
        // Guest render targets are linear (including the RGBA16F scanout used by Unity). An
        // sRGB drawable applies the display transfer function while storing into the 8-bit
        // swapchain. Presenting into BGRA8Unorm copied those linear values verbatim and crushed
        // the entire lower end of the scene toward black.
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

    pub(super) fn finalize_pending_flip(&mut self) {
        let (Some(display), Some(flip)) = (self.display, self.pending_flip) else {
            return;
        };
        match self.activate_translated_scanout(flip.address, display.width, display.height) {
            Ok(true) => {
                self.uploaded_flip_sequence = flip.sequence;
            }
            Ok(false) => {
                if let Err(error) = self.synchronize_flip_target(flip.address) {
                    eprintln!("[REMu][METAL] flip synchronization failed: {error}");
                }
            }
            Err(error) => {
                eprintln!("[REMu][METAL] translated scanout synchronization failed: {error}");
            }
        }
        self.finalized_flip_sequence = flip.sequence;
        self.redraw_pending = true;
    }

    pub(super) fn show_host_splash(&mut self, image: GpuHostImage) -> Result<(), String> {
        let device = self.device.as_ref().ok_or("Metal device is unavailable")?;
        let descriptor = TextureDescriptor::new();
        descriptor.set_texture_type(MTLTextureType::D2);
        descriptor.set_pixel_format(MTLPixelFormat::RGBA8Unorm_sRGB);
        descriptor.set_width(u64::from(image.width()));
        descriptor.set_height(u64::from(image.height()));
        descriptor.set_storage_mode(MTLStorageMode::Shared);
        descriptor.set_usage(MTLTextureUsage::ShaderRead);
        let texture = device.new_texture(&descriptor);
        texture.set_label("REMu installed-title system splash");
        let bytes_per_row = u64::from(image.width())
            .checked_mul(4)
            .ok_or("system splash row size overflows")?;
        texture.replace_region(
            MTLRegion::new_2d(0, 0, u64::from(image.width()), u64::from(image.height())),
            0,
            image.rgba8().as_ptr().cast::<c_void>(),
            bytes_per_row,
        );
        self.host_splash = Some(HostSplashTexture {
            texture,
            width: image.width(),
            height: image.height(),
        });
        self.redraw_pending = true;
        Ok(())
    }

    pub(super) fn note_pending_flip_target_write(&mut self, descriptor: GuestTextureDescriptor) {
        let (Some(display), Some(flip)) = (self.display, self.pending_flip) else {
            return;
        };
        if flip.sequence == self.uploaded_flip_sequence {
            // This flip already owns an immutable GPU snapshot. Later writes target a future
            // frame, even when the title reuses the same VideoOut allocation.
            return;
        }
        if flip.address == descriptor.address
            && display.width == descriptor.width
            && display.height == descriptor.height
        {
            // A guest submission can clear, composite, and decorate one VideoOut target through
            // several render passes. Presenting the first write exposes the clear pass and can
            // retire the flip before the interface/game image is complete. Coalesce all matching
            // writes and activate the final resident at the ordered submission boundary.
            self.pending_flip_target_written_since_boundary = true;
        }
    }

    pub(super) fn finalize_pending_flip_at_submission_boundary(
        &mut self,
        source: OrderedEventSource,
    ) {
        let initial_flip_boundary = matches!(
            (source, self.pending_flip_queue),
            (OrderedEventSource::GuestQueue(queue_id), Some(pending_queue))
                if queue_id == pending_queue
        );
        if !initial_flip_boundary && !self.pending_flip_target_written_since_boundary {
            return;
        }
        if initial_flip_boundary {
            self.pending_flip_queue = None;
        }
        self.pending_flip_target_written_since_boundary = false;
        self.finalize_pending_flip();
    }

    pub(super) fn retire_completed_presentations(&mut self) {
        loop {
            let Some(presentation) = self.pending_presentations.front() else {
                return;
            };
            match presentation.command_buffer.status() {
                MTLCommandBufferStatus::Completed => {
                    let presentation = self
                        .pending_presentations
                        .pop_front()
                        .expect("front presentation exists");
                    self.statistics.completed_presents =
                        self.statistics.completed_presents.saturating_add(1);
                    self.statistics.gpu_execution_time +=
                        completed_command_buffer_gpu_duration(&presentation.command_buffer);
                    self.statistics.presentation_queue_latency +=
                        presentation.submitted_at.elapsed();
                    if let Some(completion) = presentation.flip_completion {
                        remu_gpu::record_gpu_flip_completed(completion);
                    }
                }
                MTLCommandBufferStatus::Error => {
                    let presentation = self
                        .pending_presentations
                        .pop_front()
                        .expect("front presentation exists");
                    eprintln!("[REMu][METAL] presentation command buffer failed");
                    // A terminal presentation failure cannot be retried in place. Complete the
                    // flip after reporting it so the guest does not deadlock on an unavailable
                    // drawable while later frames may still recover.
                    if let Some(completion) = presentation.flip_completion {
                        remu_gpu::record_gpu_flip_completed(completion);
                    }
                }
                _ => return,
            }
        }
    }

    pub(super) fn update_window_title(&mut self) {
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
        let resident_mib = process_resident_bytes().map(|bytes| bytes.div_ceil(1024 * 1024));
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
        let resident =
            resident_mib.map_or_else(|| "RSS ?".to_owned(), |mib| format!("RSS {mib} MiB"));
        let pending_gpu_events = remu_gpu::gpu_work_items_pending();
        let pending_flips = remu_gpu::gpu_flips_pending();
        let title = format!(
            "REMu | FPS {flip_rate:.1} ({frame_milliseconds:.1} ms) | Present {present_rate:.1}/s | GPU {gpu_utilization:.0}% Lat {presentation_latency_milliseconds:.1} ms | {resident} | Draw {draw_rate:.0}/s RP {render_pass_rate:.0}/s CS {compute_rate:.0}/s | CB {guest_batch_rate:.0}/s PB {pipeline_barrier_rate:.0}/s ({}) | Q W{} P{} F{} | Tex {} | Pipe {}/{} | Audio {audio_output_rate:.0}/s U {audio_underrun_percent:.1}%",
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
                "[REMu][PERF-CAUSES] RP-end compute={:.0}/s key={:.0}/s flush={:.0}/s | CB tick={:.0}/s memory={:.0}/s wait={:.0}/s flip={:.0}/s capacity={:.0}/s sync={:.0}/s",
                f64::from(self.statistics.render_pass_ends_for_compute) / seconds,
                f64::from(self.statistics.render_pass_ends_for_target_change) / seconds,
                f64::from(self.statistics.render_pass_ends_for_batch_flush) / seconds,
                f64::from(self.statistics.batch_flushes_from_event_loop) / seconds,
                f64::from(self.statistics.batch_flushes_for_guest_memory_write) / seconds,
                f64::from(self.statistics.batch_flushes_for_blocked_wait) / seconds,
                f64::from(self.statistics.batch_flushes_for_flip_snapshot) / seconds,
                f64::from(self.statistics.batch_flushes_for_work_capacity) / seconds,
                f64::from(self.statistics.batch_flushes_for_cpu_synchronization) / seconds,
            );
            let compile_enqueued = PIPELINE_COMPILE_ENQUEUED.swap(0, Ordering::Relaxed);
            let compile_queue_full = PIPELINE_COMPILE_QUEUE_FULL.swap(0, Ordering::Relaxed);
            let compile_completed = PIPELINE_COMPILE_COMPLETED.swap(0, Ordering::Relaxed);
            let compile_failed = PIPELINE_COMPILE_FAILED.swap(0, Ordering::Relaxed);
            let draw_gates = PIPELINE_DRAW_GATES.swap(0, Ordering::Relaxed);
            let (pipeline_blocks, memory_blocks) = self.receiver.block_reason_counts();
            eprintln!(
                "[REMu][PERF-PIPELINE] ready={} active={} failed={} interval=enqueued:{compile_enqueued}/full:{compile_queue_full}/done:{compile_completed}/failed:{compile_failed}/gates:{draw_gates} blocked=pipeline:{pipeline_blocks}/memory:{memory_blocks}",
                self.compiled_shader_pairs.len(),
                self.compiling_shader_pairs.len(),
                self.failed_shader_pairs.len(),
            );
            let queue_diagnostics = self.receiver.queue_diagnostics();
            if !queue_diagnostics.is_empty() {
                eprintln!("[REMu][PERF-QUEUES] {queue_diagnostics}");
            }
            let snapshot_count = TEXTURE_SNAPSHOT_COUNT.swap(0, Ordering::Relaxed);
            let snapshot_bytes = TEXTURE_SNAPSHOT_BYTES.swap(0, Ordering::Relaxed);
            let snapshot_nanoseconds = TEXTURE_SNAPSHOT_NANOSECONDS.swap(0, Ordering::Relaxed);
            let decode_count = MODE27_DECODE_COUNT.swap(0, Ordering::Relaxed);
            let decode_bytes = MODE27_DECODE_BYTES.swap(0, Ordering::Relaxed);
            let decode_nanoseconds = MODE27_DECODE_NANOSECONDS.swap(0, Ordering::Relaxed);
            let upload_count = MIP_UPLOAD_COUNT.swap(0, Ordering::Relaxed);
            let upload_bytes = MIP_UPLOAD_BYTES.swap(0, Ordering::Relaxed);
            let upload_nanoseconds = MIP_UPLOAD_NANOSECONDS.swap(0, Ordering::Relaxed);
            eprintln!(
                "[REMu][PERF-TEXTURE] snapshot={snapshot_count}/{:.1}MiB/{:.1}ms mode27={decode_count}/{:.1}MiB/{:.1}ms upload={upload_count}/{:.1}MiB/{:.1}ms",
                snapshot_bytes as f64 / (1024.0 * 1024.0),
                snapshot_nanoseconds as f64 / 1_000_000.0,
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
            eprintln!(
                "[REMu][PERF-TEXTURE-CACHE] resident={} watches={} hit={} miss={} dirty_reload={} acquire_invalidate={} resident_acquire_invalidate={} capacity_evict={} depth_native={} alias_writeback={}/{:.1}MiB/{:.1}ms",
                self.sampled_texture_cache.len(),
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
        self.statistics.render_pass_ends_for_batch_flush = 0;
        self.statistics.batch_flushes_from_event_loop = 0;
        self.statistics.batch_flushes_for_guest_memory_write = 0;
        self.statistics.batch_flushes_for_blocked_wait = 0;
        self.statistics.batch_flushes_for_flip_snapshot = 0;
        self.statistics.batch_flushes_for_work_capacity = 0;
        self.statistics.batch_flushes_for_cpu_synchronization = 0;
        self.statistics.gpu_execution_time = Duration::ZERO;
        self.statistics.presentation_queue_latency = Duration::ZERO;
        self.statistics.last_audio_output_frames = audio_output;
        self.statistics.last_audio_underrun_frames = audio_underruns;
    }

    pub(super) fn synchronize_flip_target(&mut self, address: u64) -> Result<(), String> {
        let display = self.display;
        let descriptor = self
            .resident_guest_textures
            .values()
            .filter(|resident| resident.descriptor.address == address && resident.gpu_written)
            .filter(|resident| {
                display.is_none_or(|display| {
                    resident.descriptor.width == display.width
                        && resident.descriptor.height == display.height
                })
            })
            .max_by_key(|resident| resident.content_version)
            .map(|resident| resident.descriptor);
        let Some(descriptor) = descriptor else {
            return Ok(());
        };
        self.wait_for_guest_command_batches()?;
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
        let resident = self
            .resident_guest_textures
            .get_mut(&descriptor)
            .expect("selected resident texture exists");
        match refresh_resident_texture_write_watch(resident)? {
            WriteWatchRefresh::Replaced => {
                self.resident_guest_textures.remove(&descriptor);
                return Ok(());
            }
            WriteWatchRefresh::Reprotecting => return Ok(()),
            WriteWatchRefresh::Active | WriteWatchRefresh::Renewed { .. } => {}
        }
        let (alias_ranges, content_version) = {
            let resident = self
                .resident_guest_textures
                .get(&descriptor)
                .expect("refreshed resident texture exists");
            (
                resident.write_watch.alias_ranges(),
                resident.content_version,
            )
        };
        retire_sampled_textures_overlapping_writeback(
            &mut self.sampled_texture_cache,
            &alias_ranges,
        );
        retire_resident_textures_overlapping_writeback(
            &mut self.resident_guest_textures,
            &alias_ranges,
            descriptor,
            content_version,
        );
        retire_multisample_textures_overlapping_writeback(
            &mut self.multisample_textures,
            &alias_ranges,
        );
        let resident = self
            .resident_guest_textures
            .get_mut(&descriptor)
            .expect("refreshed resident texture exists");
        write_resident_texture_contents_to_guest(&device, &queue, resident)?;
        resident.gpu_written = false;
        Ok(())
    }

    pub(super) fn translated_scanout_source(
        &self,
        address: u64,
        width: u32,
        height: u32,
    ) -> Option<(GuestTextureDescriptor, Texture, &'static str)> {
        // A CB-backed resident represents the architectural render target and carries a content
        // version. A synthetic direct target is only a fallback for streams whose CB state was
        // absent, so it must never outrank newer resident compositor output at the same VideoOut
        // address.
        let resident = self
            .resident_guest_textures
            .values()
            .filter(|resident| {
                resident.descriptor.address == address
                    && resident.descriptor.width == width
                    && resident.descriptor.height == height
            })
            .max_by_key(|resident| resident.content_version)
            .map(|resident| (resident.descriptor, resident.texture.clone(), "resident"));
        let direct = self
            .direct_scanout_textures
            .get(&address)
            .filter(|scanout| {
                scanout.initialized
                    && scanout.descriptor.width == width
                    && scanout.descriptor.height == height
            })
            .map(|scanout| (scanout.descriptor, scanout.texture.clone(), "direct"));
        resident.or(direct)
    }

    pub(super) fn trace_translated_scanout_candidates(
        &self,
        address: u64,
        width: u32,
        height: u32,
        selected: GuestTextureDescriptor,
    ) {
        if !trace_scanout_writes() {
            return;
        }
        let candidates = self
            .resident_guest_textures
            .values()
            .filter(|resident| {
                resident.descriptor.address == address
                    && resident.descriptor.width == width
                    && resident.descriptor.height == height
            })
            .map(|resident| {
                (
                    resident.descriptor,
                    resident.content_version,
                    resident.gpu_written,
                )
            })
            .collect::<Vec<_>>();
        eprintln!(
            "[REMu][METAL][SCANOUT-SOURCE] address={address:#x} {}x{} selected={selected:?} candidates={candidates:?} last_writer={:?} last_texture_writer={:?}",
            width,
            height,
            diagnostic_scanout_writer(address),
            diagnostic_texture_writer(selected),
        );
    }

    pub(super) fn activate_translated_scanout(
        &mut self,
        address: u64,
        width: u32,
        height: u32,
    ) -> Result<bool, String> {
        let Some((descriptor, source_texture, source)) =
            self.translated_scanout_source(address, width, height)
        else {
            return Ok(false);
        };
        self.trace_translated_scanout_candidates(address, width, height, descriptor);
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
        let snapshot = self.ensure_scanout_snapshot_texture(&device, descriptor);
        // End and commit every encoder before the flip. The single Metal queue then orders this
        // private-to-private copy between the frame's producer work and any later command stream
        // that clears or reuses the same VideoOut allocation.
        self.flush_guest_command_batch(false, GuestBatchFlushReason::FlipSnapshot)?;
        let command_buffer = queue.new_command_buffer().to_owned();
        command_buffer.set_label("REMu ordered VideoOut snapshot");
        let blit = command_buffer.new_blit_command_encoder();
        blit.set_label("REMu translated scanout snapshot");
        blit.copy_from_texture(
            &source_texture,
            0,
            0,
            MTLOrigin { x: 0, y: 0, z: 0 },
            MTLSize::new(u64::from(descriptor.width), u64::from(descriptor.height), 1),
            &snapshot,
            0,
            0,
            MTLOrigin { x: 0, y: 0, z: 0 },
        );
        blit.end_encoding();
        command_buffer.commit();
        self.frame_texture = Some(snapshot);
        self.frame_texture_size = (descriptor.width, descriptor.height);
        self.active_translated_scanout = Some(ActiveTranslatedScanout {
            address,
            width,
            height,
            sequence: self
                .pending_flip
                .filter(|flip| flip.address == address)
                .map_or(self.next_flip_sequence, |flip| flip.sequence),
        });
        static LOGGED_TRANSLATED_SCANOUTS: OnceLock<Mutex<HashSet<(u64, &'static str)>>> =
            OnceLock::new();
        let first_for_source = LOGGED_TRANSLATED_SCANOUTS
            .get_or_init(|| Mutex::new(HashSet::new()))
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .insert((address, source));
        if first_for_source {
            eprintln!(
                "[REMu][METAL] snapshotted translated {source} scanout sequence={} at {address:#x} {}x{} format={:?} tile={}",
                self.next_flip_sequence,
                descriptor.width,
                descriptor.height,
                descriptor.pixel_format,
                descriptor.tile_mode,
            );
        }
        Ok(true)
    }

    pub(super) fn upload_pending_frame(&mut self) {
        let (Some(display), Some(flip)) = (self.display, self.pending_flip) else {
            return;
        };
        if flip.sequence == self.uploaded_flip_sequence {
            return;
        }
        match self.activate_translated_scanout(flip.address, display.width, display.height) {
            Ok(true) => {
                self.uploaded_flip_sequence = flip.sequence;
                return;
            }
            Ok(false) => {}
            Err(error) => {
                eprintln!("[REMu][METAL] translated scanout retry failed: {error}");
            }
        }
        // A translated render target is authoritative when one exists. Before the first GPU
        // render pass, however, splash screens and decoded videos can be written directly into a
        // registered VideoOut allocation. Preserve both ABI layouts here: zero is the console's
        // display-tiled mode (the same mode-27 layout exposed by its color target), while one is
        // row-linear.
        let Some(guest_descriptor) = raw_video_out_texture_descriptor(flip.address, display) else {
            if !self.logged_unsupported_layout {
                eprintln!(
                    "[REMu][METAL] cannot upload display layout tiling={} pitch={} width={} yet",
                    display.tiling_mode, display.pitch, display.width
                );
                self.logged_unsupported_layout = true;
            }
            return;
        };
        let Some(device) = self.device.as_ref() else {
            return;
        };
        let Some(row_bytes) = u64::from(display.width).checked_mul(4) else {
            return;
        };
        if flip.address == 0 {
            return;
        }
        let frame = match snapshot_guest_texture_raw(guest_descriptor)
            .and_then(|raw| decode_guest_texture_linear_from_raw(guest_descriptor, &raw))
        {
            Ok(frame) => frame,
            Err(error) => {
                if !self.logged_unsupported_layout {
                    eprintln!(
                        "[REMu][METAL] could not upload raw display buffer at {:#x}: {error}",
                        flip.address
                    );
                    self.logged_unsupported_layout = true;
                }
                return;
            }
        };
        if self.frame_texture_size != (display.width, display.height) {
            let descriptor = TextureDescriptor::new();
            descriptor.set_texture_type(MTLTextureType::D2);
            descriptor.set_pixel_format(MTLPixelFormat::BGRA8Unorm);
            descriptor.set_width(u64::from(display.width));
            descriptor.set_height(u64::from(display.height));
            descriptor.set_storage_mode(MTLStorageMode::Shared);
            descriptor.set_usage(MTLTextureUsage::ShaderRead);
            self.frame_texture = Some(device.new_texture(&descriptor));
            self.frame_texture_size = (display.width, display.height);
        }
        let Some(texture) = self.frame_texture.as_ref() else {
            return;
        };
        let Ok(bytes_per_row) = usize::try_from(row_bytes) else {
            return;
        };
        texture.replace_region(
            MTLRegion::new_2d(0, 0, u64::from(display.width), u64::from(display.height)),
            0,
            frame.as_ptr().cast::<c_void>(),
            bytes_per_row as u64,
        );
        self.active_translated_scanout = None;
        let capture = (self.capture_directory.is_some()
            && self
                .last_guest_capture
                .is_none_or(|last| last.elapsed() >= capture_interval()))
        .then(|| {
            let descriptor = GuestTextureDescriptor {
                address: flip.address,
                width: display.width,
                height: display.height,
                pitch: display.width,
                tile_mode: 0,
                pixel_format: MTLPixelFormat::BGRA8Unorm,
                bytes_per_pixel: 4,
            };
            self.command_queue
                .as_ref()
                .ok_or_else(|| "Metal command queue is unavailable".to_owned())
                .and_then(|queue| read_texture_linear(device, queue, texture, descriptor))
                .map(|bytes| (descriptor, bytes))
        });
        self.uploaded_flip_sequence = flip.sequence;
        if flip.sequence == 1 {
            eprintln!(
                "[REMu][METAL] uploaded first guest display buffer format={:#x}",
                display.pixel_format
            );
        }
        if let Some(capture) = capture {
            match capture {
                Ok((descriptor, bgra)) => {
                    self.guest_bgra_bytes = bgra;
                    self.capture_guest_frame_if_changed(descriptor);
                }
                Err(error) => eprintln!("[REMu][METAL] could not read back flipped frame: {error}"),
            }
        }
    }

    pub(super) fn update_keyboard_input(&mut self, code: KeyCode, state: ElementState) {
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

    pub(super) fn publish_pad_state(&self) {
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

    pub(super) fn render(&mut self) {
        if !self.redraw_pending {
            return;
        }
        self.upload_pending_frame();
        let translated_snapshot = self
            .active_translated_scanout
            .filter(|scanout| scanout.sequence == self.uploaded_flip_sequence)
            .and_then(|scanout| {
                self.scanout_snapshot_textures
                    .get(&scanout.address)
                    .filter(|snapshot| {
                        snapshot.descriptor.width == scanout.width
                            && snapshot.descriptor.height == scanout.height
                    })
                    .map(|snapshot| (snapshot.descriptor, snapshot.texture.clone()))
            });
        let guest_frame_texture = self.frame_texture.clone();
        let host_splash = self
            .host_splash
            .as_ref()
            .map(|splash| (splash.texture.clone(), splash.width, splash.height));
        let (frame_texture, source_dimensions) = if let Some((texture, width, height)) = host_splash
        {
            (Some(texture), Some((width, height)))
        } else {
            (
                guest_frame_texture,
                self.display.map(|display| (display.width, display.height)),
            )
        };
        let capture_due = self.capture_directory.is_some()
            && self
                .last_guest_capture
                .is_none_or(|last| last.elapsed() >= capture_interval());
        if capture_due
            && let Some((descriptor, texture)) = translated_snapshot.as_ref()
            && matches!(
                descriptor.pixel_format,
                MTLPixelFormat::RGBA8Unorm
                    | MTLPixelFormat::RGBA8Unorm_sRGB
                    | MTLPixelFormat::BGRA8Unorm
                    | MTLPixelFormat::BGRA8Unorm_sRGB
                    | MTLPixelFormat::RGBA16Float
            )
            && let (Some(device), Some(queue)) = (
                self.device.as_ref().cloned(),
                self.command_queue.as_ref().cloned(),
            )
        {
            match read_texture_linear(&device, &queue, texture, *descriptor) {
                Ok(rendered) => self.capture_translated_frame_if_changed(*descriptor, &rendered),
                Err(error) => {
                    eprintln!("[REMu][METAL] could not capture live translated frame: {error}");
                }
            }
        }
        // A host splash is presentable before VideoOut registration. After HideSplash, a missing
        // guest texture intentionally clears the old system-owned image to black until the first
        // guest scanout is ready.
        let (Some(layer), Some(queue)) = (self.layer.as_ref(), self.command_queue.as_ref()) else {
            return;
        };
        let Some(drawable) = layer.next_drawable() else {
            return;
        };
        let command_buffer = queue.new_command_buffer().to_owned();
        command_buffer.set_label("REMu guest presentation");
        let Ok(destination_width) = u32::try_from(drawable.texture().width()) else {
            return;
        };
        let Ok(destination_height) = u32::try_from(drawable.texture().height()) else {
            return;
        };
        let pass = RenderPassDescriptor::new();
        let attachment = pass
            .color_attachments()
            .object_at(0)
            .expect("Metal presentation pass has color attachment zero");
        attachment.set_texture(Some(drawable.texture()));
        attachment.set_load_action(MTLLoadAction::Clear);
        attachment.set_store_action(MTLStoreAction::Store);
        attachment.set_clear_color(MTLClearColor::new(0.0, 0.0, 0.0, 1.0));
        let encoder = command_buffer.new_render_command_encoder(pass);
        encoder.set_label("REMu scaled display-buffer presentation");
        if let Some(texture) = frame_texture.as_ref() {
            let presentation = self
                .presentation
                .as_ref()
                .expect("windowed presenter has presentation resources");
            let (source_width, source_height) = source_dimensions
                .expect("every presentation texture has validated source dimensions");
            encoder.set_render_pipeline_state(&presentation.pipeline);
            encoder.set_viewport(presentation_viewport(
                source_width,
                source_height,
                destination_width,
                destination_height,
            ));
            encoder.set_fragment_texture(0, Some(texture));
            encoder.set_fragment_sampler_state(0, Some(&presentation.sampler));
            encoder.draw_primitives(MTLPrimitiveType::Triangle, 0, 3);
        }
        encoder.end_encoding();
        command_buffer.present_drawable(drawable);
        command_buffer.commit();
        let flip_completion = self
            .pending_flip
            .filter(|flip| {
                flip.sequence == self.finalized_flip_sequence
                    && flip.sequence > self.submitted_flip_sequence
            })
            .map(|flip| {
                self.submitted_flip_sequence = flip.sequence;
                remu_gpu::GpuFlipCompletion {
                    address: flip.address,
                    buffer_index: flip.buffer_index,
                    flip_argument: flip.flip_argument,
                }
            });
        self.pending_presentations.push_back(PendingPresentation {
            command_buffer,
            submitted_at: Instant::now(),
            flip_completion,
        });
        self.redraw_pending = false;
    }
}
pub(super) fn headless_metal_value_is_enabled(value: Option<&OsStr>) -> bool {
    value.is_some_and(|value| value == "1")
}

pub(super) fn headless_metal_requested() -> bool {
    headless_metal_value_is_enabled(std::env::var_os("REMU_HEADLESS_METAL").as_deref())
}

pub(super) const PRESENTATION_SHADER_SOURCE: &str = r"
#include <metal_stdlib>
using namespace metal;

struct REmuPresentationVertex {
    float4 position [[position]];
    float2 texture_coordinate;
};

vertex REmuPresentationVertex remu_presentation_vertex(uint vertex_id [[vertex_id]]) {
    constexpr float2 positions[] = {
        float2(-1.0, -1.0),
        float2( 3.0, -1.0),
        float2(-1.0,  3.0),
    };
    constexpr float2 texture_coordinates[] = {
        float2(0.0,  1.0),
        float2(2.0,  1.0),
        float2(0.0, -1.0),
    };
    return {
        float4(positions[vertex_id], 0.0, 1.0),
        texture_coordinates[vertex_id],
    };
}

fragment float4 remu_presentation_fragment(
    REmuPresentationVertex input [[stage_in]],
    texture2d<float> source [[texture(0)]],
    sampler source_sampler [[sampler(0)]]) {
    // VideoOut treats scanout alpha as padding; games are free to leave it at zero. Passing that
    // value into CAMetalLayer makes AppKit composite an otherwise valid frame as transparent
    // black, so host presentation must always be opaque.
    return float4(source.sample(source_sampler, input.texture_coordinate).rgb, 1.0);
}
";

impl PresentationResolution {
    pub(super) fn from_environment() -> Result<Self, String> {
        let Some(value) = std::env::var_os("REMU_PRESENTATION_SIZE") else {
            return Ok(Self::LogicalWindow);
        };
        let value = value
            .into_string()
            .map_err(|_| "REMU_PRESENTATION_SIZE is not valid UTF-8".to_owned())?;
        parse_presentation_resolution(&value)
    }
}

pub(super) fn create_presentation_resources(
    device: &Device,
) -> Result<PresentationResources, String> {
    let library = if std::env::var_os("REMU_METAL_COMPILER").is_some() {
        compile_metal_library_with_tools(device, "presentation", PRESENTATION_SHADER_SOURCE)?
    } else {
        let options = CompileOptions::new();
        options.set_fast_math_enabled(false);
        device
            .new_library_with_source(PRESENTATION_SHADER_SOURCE, &options)
            .map_err(|error| format!("Metal presentation shader compilation: {error}"))?
    };
    let vertex_function = library
        .get_function("remu_presentation_vertex", None)
        .map_err(|error| format!("Metal presentation vertex function: {error}"))?;
    let fragment_function = library
        .get_function("remu_presentation_fragment", None)
        .map_err(|error| format!("Metal presentation fragment function: {error}"))?;
    let descriptor = RenderPipelineDescriptor::new();
    descriptor.set_label("REMu scaled display presentation");
    descriptor.set_vertex_function(Some(&vertex_function));
    descriptor.set_fragment_function(Some(&fragment_function));
    descriptor
        .color_attachments()
        .object_at(0)
        .ok_or("Metal has no presentation color attachment")?
        .set_pixel_format(MTLPixelFormat::BGRA8Unorm_sRGB);
    let pipeline = device
        .new_render_pipeline_state(&descriptor)
        .map_err(|error| format!("Metal presentation pipeline: {error}"))?;
    let sampler_descriptor = SamplerDescriptor::new();
    sampler_descriptor.set_address_mode_s(MTLSamplerAddressMode::ClampToEdge);
    sampler_descriptor.set_address_mode_t(MTLSamplerAddressMode::ClampToEdge);
    sampler_descriptor.set_min_filter(MTLSamplerMinMagFilter::Linear);
    sampler_descriptor.set_mag_filter(MTLSamplerMinMagFilter::Linear);
    sampler_descriptor.set_mip_filter(MTLSamplerMipFilter::NotMipmapped);
    let sampler = device.new_sampler(&sampler_descriptor);
    Ok(PresentationResources {
        pipeline,
        _vertex_function: vertex_function,
        _fragment_function: fragment_function,
        sampler,
    })
}

pub(super) fn presenter_window_attributes() -> winit::window::WindowAttributes {
    Window::default_attributes()
        .with_title("REmu")
        .with_inner_size(LogicalSize::new(1280, 720))
        .with_active(true)
}

pub(super) fn parse_presentation_resolution(value: &str) -> Result<PresentationResolution, String> {
    let normalized = value.trim().to_ascii_lowercase();
    match normalized.as_str() {
        "window" | "logical" => Ok(PresentationResolution::LogicalWindow),
        "native" => Ok(PresentationResolution::NativeWindow),
        _ => {
            let (width, height) = normalized.split_once('x').ok_or_else(|| {
                format!(
                    "invalid REMU_PRESENTATION_SIZE {value:?}; expected window, native, or WIDTHxHEIGHT"
                )
            })?;
            let width = width
                .parse::<u32>()
                .map_err(|_| format!("invalid width in REMU_PRESENTATION_SIZE {value:?}"))?;
            let height = height
                .parse::<u32>()
                .map_err(|_| format!("invalid height in REMU_PRESENTATION_SIZE {value:?}"))?;
            if width == 0 || height == 0 || width > 16_384 || height > 16_384 {
                return Err(format!(
                    "REMU_PRESENTATION_SIZE dimensions must each be in 1..=16384, got {width}x{height}"
                ));
            }
            Ok(PresentationResolution::Fixed(width, height))
        }
    }
}

#[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
pub(super) fn presentation_drawable_size(
    resolution: PresentationResolution,
    physical_width: u32,
    physical_height: u32,
    scale_factor: f64,
) -> (u32, u32) {
    match resolution {
        PresentationResolution::Fixed(width, height) => (width, height),
        PresentationResolution::NativeWindow => (physical_width.max(1), physical_height.max(1)),
        PresentationResolution::LogicalWindow => {
            let scale_factor = if scale_factor.is_finite() && scale_factor > 0.0 {
                scale_factor
            } else {
                1.0
            };
            let logical = |physical: u32| {
                // The result is rounded and clamped to the complete accepted u32 subrange before
                // conversion, so neither truncation nor a negative result is possible.
                (f64::from(physical) / scale_factor)
                    .round()
                    .clamp(1.0, 16_384.0) as u32
            };
            (logical(physical_width), logical(physical_height))
        }
    }
}

pub(super) fn presentation_viewport(
    source_width: u32,
    source_height: u32,
    destination_width: u32,
    destination_height: u32,
) -> MTLViewport {
    let source_width = f64::from(source_width.max(1));
    let source_height = f64::from(source_height.max(1));
    let destination_width = f64::from(destination_width.max(1));
    let destination_height = f64::from(destination_height.max(1));
    let scale = (destination_width / source_width).min(destination_height / source_height);
    let width = source_width * scale;
    let height = source_height * scale;
    MTLViewport {
        originX: (destination_width - width) * 0.5,
        originY: (destination_height - height) * 0.5,
        width,
        height,
        znear: 0.0,
        zfar: 1.0,
    }
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
        if let Err(error) = self.flush_guest_command_batch(false, GuestBatchFlushReason::EventLoop)
        {
            eprintln!("[REMu][METAL] guest command batch submission failed: {error}");
        }
        self.retire_completed_presentations();
        self.update_window_title();
        if self.redraw_pending
            && let Some(window) = self.window.as_ref()
        {
            window.request_redraw();
        }
        // The std channel feeding guest GPU work cannot wake Winit directly. A short deadline
        // keeps submission latency low without ControlFlow::Poll busy-spinning an otherwise idle
        // process or continuously presenting identical drawables.
        event_loop.set_control_flow(ControlFlow::WaitUntil(Instant::now() + EVENT_POLL_INTERVAL));
    }
}

pub(super) fn run_headless_presenter(receiver: Receiver<GpuEvent>) -> Result<(), String> {
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
                presenter.flush_guest_command_batch(false, GuestBatchFlushReason::EventLoop)
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
