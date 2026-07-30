use std::io::Read;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{self, SyncSender, TryRecvError, TrySendError};
use std::sync::{Arc, OnceLock};
use std::time::{Duration, Instant};

use crate::{
    AvPlayerDecoder, AvPlayerEventCallback, AvPlayerEventDelivery, AvPlayerInstance,
    AvPlayerTextureDeallocation, PROBE_DISPATCH, ProbeDispatchState, current_guest_thread_handle,
    current_guest_thread_name, guest_aligned_alloc, guest_memset,
    process_guest_range_has_read_access, process_guest_range_is_readable, read_guest_cstring,
    read_guest_u8, read_guest_u32, read_guest_u64, resolve_guest_path, sce_kernel_error,
    write_guest_u8, write_guest_u16, write_guest_u32, write_guest_u64,
};

pub(crate) const AV_PLAYER_ERROR_INVALID_PARAMS: u32 = 0x806a_0001;
pub(crate) const AV_PLAYER_ERROR_OPERATION_FAILED: u32 = 0x806a_0002;

#[derive(Debug, PartialEq, Eq)]
enum AvPlayerDecoderPoll {
    Frame(Vec<u8>),
    Pending,
    Finished,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum AvPlayerFrameStatus {
    Ready,
    Pending,
    Finished,
}

fn av_player_decoder_reaper() -> &'static mpsc::Sender<AvPlayerDecoder> {
    static REAPER: OnceLock<mpsc::Sender<AvPlayerDecoder>> = OnceLock::new();
    REAPER.get_or_init(|| {
        let (sender, receiver) = mpsc::channel::<AvPlayerDecoder>();
        std::thread::Builder::new()
            .name("REMu AVPlayer reaper".to_owned())
            .spawn(move || {
                while let Ok(mut decoder) = receiver.recv() {
                    decoder.reap();
                }
            })
            .expect("AVPlayer decoder reaper thread must start");
        sender
    })
}

impl AvPlayerDecoder {
    fn poll(&self) -> AvPlayerDecoderPoll {
        let frames = self
            .frames
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        match frames.try_recv() {
            Ok(frame) => AvPlayerDecoderPoll::Frame(frame),
            Err(TryRecvError::Empty) => AvPlayerDecoderPoll::Pending,
            Err(TryRecvError::Disconnected) => AvPlayerDecoderPoll::Finished,
        }
    }

    pub(crate) fn retire(mut self) {
        self.cancelled.store(true, Ordering::Release);
        if let Some(child) = self.child.as_mut() {
            // Closing the process side of the pipe makes a decoder worker blocked in read_exact
            // return immediately. Child collection and the host-thread join happen on the
            // reaper, never under the process-wide guest dispatch lock.
            let _ = child.kill();
        }
        if let Err(error) = av_player_decoder_reaper().send(self) {
            let mut decoder = error.0;
            decoder.reap();
        }
    }

    fn reap(&mut self) {
        self.cancelled.store(true, Ordering::Release);
        if let Some(mut child) = self.child.take() {
            let _ = child.kill();
            let _ = child.wait();
        }
        if let Some(worker) = self.worker.take() {
            let _ = worker.join();
        }
    }
}

impl Drop for AvPlayerDecoder {
    fn drop(&mut self) {
        // Normal playback teardown reaches `retire` and leaves both handles empty before drop.
        // This fallback is for a failed reaper startup/channel and process-state destruction.
        if self.child.is_some() || self.worker.is_some() {
            self.reap();
        }
    }
}

fn av_player_send_decoded_frame(
    sender: &SyncSender<Vec<u8>>,
    cancelled: &AtomicBool,
    mut frame: Vec<u8>,
) -> bool {
    loop {
        if cancelled.load(Ordering::Acquire) {
            return false;
        }
        match sender.try_send(frame) {
            Ok(()) => return true,
            Err(TrySendError::Full(returned)) => {
                frame = returned;
                std::thread::sleep(Duration::from_millis(1));
            }
            Err(TrySendError::Disconnected(_)) => return false,
        }
    }
}

fn av_player_decoder_worker(
    mut output: std::process::ChildStdout,
    frame_bytes: usize,
    sender: SyncSender<Vec<u8>>,
    cancelled: Arc<AtomicBool>,
) {
    while !cancelled.load(Ordering::Acquire) {
        let mut frame = vec![0_u8; frame_bytes];
        if output.read_exact(&mut frame).is_err()
            || !av_player_send_decoded_frame(&sender, &cancelled, frame)
        {
            break;
        }
    }
}

pub(crate) fn av_player_tool(name: &str, environment: &str) -> Option<PathBuf> {
    if let Some(configured) = std::env::var_os(environment).map(PathBuf::from)
        && configured.is_file()
    {
        return Some(configured);
    }
    std::env::var_os("PATH")
        .into_iter()
        .flat_map(|path| std::env::split_paths(&path).collect::<Vec<_>>())
        .map(|directory| directory.join(name))
        .chain(
            ["/opt/homebrew/bin", "/usr/local/bin", "/usr/bin"]
                .into_iter()
                .map(Path::new)
                .map(|directory| directory.join(name)),
        )
        .find(|candidate| candidate.is_file())
}

fn av_player_probe_video(path: &Path) -> Option<(u32, u32, f64, Duration)> {
    let ffprobe = av_player_tool("ffprobe", "REMU_FFPROBE")?;
    let output = Command::new(ffprobe)
        .args([
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=width,height,avg_frame_rate,duration",
            "-of",
            "default=noprint_wrappers=1",
        ])
        .arg(path)
        .output()
        .ok()?;
    if !output.status.success() {
        return None;
    }
    let fields = String::from_utf8(output.stdout).ok()?;
    let mut width = 0;
    let mut height = 0;
    let mut frames_per_second = 30.0;
    let mut duration = Duration::ZERO;
    for line in fields.lines() {
        let Some((name, value)) = line.split_once('=') else {
            continue;
        };
        match name {
            "width" => width = value.parse().ok()?,
            "height" => height = value.parse().ok()?,
            "avg_frame_rate" => {
                let (numerator, denominator) = value.split_once('/').unwrap_or((value, "1"));
                let (Ok(numerator), Ok(denominator)) =
                    (numerator.parse::<f64>(), denominator.parse::<f64>())
                else {
                    continue;
                };
                if numerator.is_finite() && denominator.is_finite() && denominator > 0.0 {
                    frames_per_second = numerator / denominator;
                }
            }
            "duration" => {
                if let Ok(seconds) = value.parse::<f64>()
                    && seconds.is_finite()
                    && seconds >= 0.0
                {
                    duration = Duration::from_secs_f64(seconds);
                }
            }
            _ => {}
        }
    }
    (width != 0 && height != 0 && frames_per_second.is_finite() && frames_per_second > 0.0)
        .then_some((width, height, frames_per_second, duration))
}

fn av_player_probe_has_audio(path: &Path) -> bool {
    let Some(ffprobe) = av_player_tool("ffprobe", "REMU_FFPROBE") else {
        return false;
    };
    Command::new(ffprobe)
        .args([
            "-v",
            "error",
            "-select_streams",
            "a:0",
            "-show_entries",
            "stream=index",
            "-of",
            "csv=p=0",
        ])
        .arg(path)
        .output()
        .is_ok_and(|output| output.status.success() && !output.stdout.is_empty())
}

/// Clamps a media source path at the application filesystem root.
///
/// AVPlayer resolves sources inside the mounted application filesystem, which has no parent above
/// its root. A source beginning with `.` or `..` segments names a directory outside that root, so
/// the segments are dropped and the remainder resolves from the root, matching how a sandboxed
/// filesystem clamps an escape at its mount point. A bare filename left after one or more parent
/// segments has no directory left to anchor it, so it is refused rather than silently becoming a
/// root lookup.
///
/// A path with no leading dot segments is returned unchanged.
fn av_player_clamp_leading_dot_segments(path: &str) -> Option<&str> {
    let mut normalized = path;
    let mut removed_parent = false;
    loop {
        if let Some(rest) = normalized.strip_prefix("../") {
            normalized = rest;
            removed_parent = true;
        } else if let Some(rest) = normalized.strip_prefix("./") {
            normalized = rest;
        } else {
            break;
        }
    }
    (!removed_parent || normalized.contains('/')).then_some(normalized)
}

pub(crate) fn av_player_resolve_source(game_root: &Path, source: &str) -> Option<PathBuf> {
    let mut normalized = source.trim().trim_end_matches('\0').replace('\\', "/");
    let scheme_length = if normalized
        .get(.."file://".len())
        .is_some_and(|scheme| scheme.eq_ignore_ascii_case("file://"))
    {
        Some("file://".len())
    } else if normalized
        .get(.."file:".len())
        .is_some_and(|scheme| scheme.eq_ignore_ascii_case("file:"))
    {
        Some("file:".len())
    } else {
        None
    };
    if let Some(scheme_length) = scheme_length {
        normalized = normalized[scheme_length..].to_owned();
    }
    // A `file:` source carries a filesystem-relative path once its scheme is stripped, so both
    // forms reach the resolver already clamped to the application root.
    let normalized = av_player_clamp_leading_dot_segments(&normalized)?.to_owned();
    resolve_guest_path(game_root, &normalized).or_else(|| {
        let media = normalized.find("Media/")?;
        resolve_guest_path(game_root, &format!("/app0/{}", &normalized[media..]))
    })
}

pub(crate) fn av_player_set_log_callback(
    arguments: [u64; 6],
    state: &mut ProbeDispatchState,
) -> u64 {
    state.av_player.log_callback = (arguments[0] != 0).then_some((arguments[0], arguments[1]));
    0
}

pub(crate) fn av_player_init_ex(arguments: [u64; 6], state: &mut ProbeDispatchState) -> u64 {
    const MINIMUM_INITIALIZATION_SIZE: usize = 48;
    const MAXIMUM_INITIALIZATION_SIZE: usize = 0x400;
    let (parameter, output) = (arguments[0], arguments[1]);
    if parameter == 0 || output == 0 {
        eprintln!(
            "[REMu][AVPLAYER] InitEx rejected null parameter={parameter:#x} output={output:#x}"
        );
        return sce_kernel_error(AV_PLAYER_ERROR_INVALID_PARAMS);
    }
    if !process_guest_range_has_read_access(parameter, std::mem::size_of::<u64>())
        || !process_guest_range_has_read_access(output, std::mem::size_of::<u64>())
    {
        eprintln!(
            "[REMu][AVPLAYER] InitEx rejected inaccessible parameter={parameter:#x} output={output:#x}"
        );
        return sce_kernel_error(AV_PLAYER_ERROR_INVALID_PARAMS);
    }
    let raw_size = read_guest_u64(parameter);
    let Ok(size) = usize::try_from(raw_size) else {
        eprintln!("[REMu][AVPLAYER] InitEx rejected non-host size={raw_size:#x}");
        return sce_kernel_error(AV_PLAYER_ERROR_INVALID_PARAMS);
    };
    if !(MINIMUM_INITIALIZATION_SIZE..=MAXIMUM_INITIALIZATION_SIZE).contains(&size) {
        eprintln!(
            "[REMu][AVPLAYER] InitEx rejected size={size:#x} parameter={parameter:#x} output={output:#x}"
        );
        return sce_kernel_error(AV_PLAYER_ERROR_INVALID_PARAMS);
    }
    if !process_guest_range_has_read_access(parameter, size) {
        eprintln!(
            "[REMu][AVPLAYER] InitEx rejected unreadable initialization parameter={parameter:#x} size={size:#x}"
        );
        return sce_kernel_error(AV_PLAYER_ERROR_INVALID_PARAMS);
    }
    let allocator_callbacks = [16_u64, 24, 32, 40].map(|offset| read_guest_u64(parameter + offset));
    if allocator_callbacks.contains(&0) {
        eprintln!(
            "[REMu][AVPLAYER] InitEx rejected allocator callbacks={allocator_callbacks:#x?} size={size:#x}"
        );
        return sce_kernel_error(AV_PLAYER_ERROR_INVALID_PARAMS);
    }
    // SAFETY: the leading bounded size field describes a caller-owned initialization structure.
    let initialization =
        unsafe { std::slice::from_raw_parts(parameter as *const u8, size).to_vec() };
    let handle = state.av_player.next_handle.max(1);
    state.av_player.next_handle = handle.saturating_add(1);
    state.av_player.players.insert(
        handle,
        AvPlayerInstance {
            _initialization: initialization,
            allocator_object: read_guest_u64(parameter + 8),
            allocate_texture_callback: read_guest_u64(parameter + 32),
            deallocate_texture_callback: read_guest_u64(parameter + 40),
            event_object: (size >= 96)
                .then(|| read_guest_u64(parameter + 88))
                .unwrap_or(0),
            event_callback: (size >= 104)
                .then(|| read_guest_u64(parameter + 96))
                .unwrap_or(0),
            source: None,
            // Gen5 InitDataEx places autoStart after the event replacement, language pointer,
            // and debug level. Offset 0xa8 belongs to the first decoder-thread record; treating
            // its priority byte as autoStart spuriously double-started explicit players.
            auto_start: size > 116 && read_guest_u8(parameter + 116) != 0,
            width: 0,
            height: 0,
            frames_per_second: 30.0,
            duration: Duration::ZERO,
            has_audio: false,
            active_session: false,
            paused: false,
            end_of_stream: false,
            playing_since: None,
            elapsed: Duration::ZERO,
            looping: false,
            decoder_seek: Duration::ZERO,
            trick_speed_percent: 100,
            video_decoder: None,
            raw_video_frame: Vec::new(),
            padded_video_frame: Vec::new(),
            video_buffer_base: 0,
            video_buffers: [0; 3],
            video_buffer_stride: 0,
            next_video_buffer: 0,
            next_video_frame: 0,
            video_catch_up_origin: None,
            audio_decoder: None,
            raw_audio_frame: Vec::new(),
            audio_buffer_base: 0,
            next_audio_buffer: 0,
            next_audio_frame: 0,
        },
    );
    write_guest_u64(output, handle);
    let player = &state.av_player.players[&handle];
    eprintln!(
        "[REMu][AVPLAYER] initialized handle={handle:#x} size={size:#x} texture_alloc={:#x}/{:#x} event={:#x}/{:#x}",
        player.allocator_object,
        player.allocate_texture_callback,
        player.event_object,
        player.event_callback
    );
    0
}

pub(crate) fn av_player_post_init(arguments: [u64; 6], state: &ProbeDispatchState) -> u64 {
    let (handle, parameter) = (arguments[0], arguments[1]);
    if parameter == 0 || !state.av_player.players.contains_key(&handle) {
        return sce_kernel_error(AV_PLAYER_ERROR_INVALID_PARAMS);
    }
    eprintln!("[REMu][AVPLAYER] post-initialized handle={handle:#x}");
    0
}

fn av_player_queue_event(
    function: u64,
    object: u64,
    event: u64,
    delivery: AvPlayerEventDelivery,
    state: &mut ProbeDispatchState,
) {
    if function == 0 {
        eprintln!("[REMu][AVPLAYER] skipped event={event} without callback");
        return;
    }
    state
        .av_player
        .pending_callbacks
        .push(AvPlayerEventCallback {
            function,
            object,
            event,
            delivery,
        });
}

pub(crate) fn take_av_player_calling_thread_callbacks(
    state: &mut ProbeDispatchState,
) -> Vec<AvPlayerEventCallback> {
    let mut calling_thread = Vec::new();
    state.av_player.pending_callbacks.retain(|callback| {
        if callback.delivery == AvPlayerEventDelivery::CallingThread {
            calling_thread.push(*callback);
            false
        } else {
            true
        }
    });
    calling_thread
}

pub(crate) fn invoke_av_player_event_callbacks(callbacks: Vec<AvPlayerEventCallback>) {
    for callback in callbacks {
        if !process_guest_range_is_readable(callback.function, 1) {
            eprintln!(
                "[REMu][AVPLAYER] rejected unreadable event={} callback={:#x}",
                callback.event, callback.function
            );
            continue;
        }
        type Callback = unsafe extern "C" fn(u64, u64, u64, u64);
        // SAFETY: InitEx supplies a live guest SysV event callback and opaque event object. The
        // process-wide dispatch lock has already been released, so this callback may re-enter any
        // AVPlayer export exactly as the native library permits.
        let function: Callback = unsafe { std::mem::transmute(callback.function) };
        // SAFETY: The callback ABI is `(object, event, source_id, event_data)`.
        unsafe { function(callback.object, callback.event, 0, 0) };
        eprintln!(
            "[REMu][AVPLAYER] delivered event={} callback={:#x} thread={:?}/{:#x}",
            callback.event,
            callback.function,
            current_guest_thread_name(),
            current_guest_thread_handle(),
        );
    }
}

pub(crate) fn invoke_av_player_texture_deallocations(
    deallocations: Vec<AvPlayerTextureDeallocation>,
) {
    type DeallocateTexture = unsafe extern "C" fn(u64, u64);
    for deallocation in deallocations {
        if deallocation.address == 0 {
            continue;
        }
        if !process_guest_range_has_read_access(deallocation.function, 1) {
            eprintln!(
                "[REMu][AVPLAYER] leaked texture={:#x}: deallocator callback={:#x} is unreadable",
                deallocation.address, deallocation.function
            );
            continue;
        }
        // SAFETY: InitEx supplied the live texture-deallocation callback and allocator object.
        // Callbacks are invoked only after releasing the process dispatch lock, so the guest
        // allocator may re-enter kernel memory imports exactly as it can in the native library.
        let function: DeallocateTexture = unsafe { std::mem::transmute(deallocation.function) };
        unsafe { function(deallocation.object, deallocation.address) };
    }
}

pub(crate) fn prepare_av_player_video_buffers(handle: u64, extended: bool) {
    let Some(dispatch) = PROBE_DISPATCH.get() else {
        return;
    };
    let request = {
        let state = dispatch
            .read()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let Some(player) = state.av_player.players.get(&handle) else {
            return;
        };
        if player.video_buffers[0] != 0
            || player.allocate_texture_callback == 0
            || player.width == 0
            || player.height == 0
        {
            return;
        }
        let Some((_, stride)) = av_player_nv12_layout(player.width, player.height, extended) else {
            return;
        };
        (
            player.allocate_texture_callback,
            player.deallocate_texture_callback,
            player.allocator_object,
            stride,
        )
    };
    if !process_guest_range_has_read_access(request.0, 1)
        || !process_guest_range_has_read_access(request.1, 1)
    {
        eprintln!(
            "[REMu][AVPLAYER] texture allocator callbacks are unreadable: allocate={:#x} deallocate={:#x}",
            request.0, request.1
        );
        return;
    }
    type AllocateTexture = unsafe extern "C" fn(u64, u32, u32) -> u64;
    // SAFETY: InitEx supplies the live guest texture-allocation callback and opaque allocator
    // object. This runs before the process dispatch lock is acquired, so the allocator may call
    // kernel memory exports while it obtains GPU-visible storage.
    let allocate: AllocateTexture = unsafe { std::mem::transmute(request.0) };
    let mut buffers = [0_u64; 3];
    for (index, buffer) in buffers.iter_mut().enumerate() {
        let Ok(size) = u32::try_from(request.3) else {
            return;
        };
        // SAFETY: The callback ABI is `(object, alignment, size)`.
        *buffer = unsafe { allocate(request.2, 0x100, size) };
        if *buffer == 0 {
            eprintln!("[REMu][AVPLAYER] texture allocator returned null for buffer {index}");
            invoke_av_player_texture_deallocations(
                buffers[..index]
                    .iter()
                    .copied()
                    .map(|address| AvPlayerTextureDeallocation {
                        function: request.1,
                        object: request.2,
                        address,
                    })
                    .collect(),
            );
            return;
        }
    }
    let mut state = dispatch
        .write()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let installed = if let Some(player) = state.av_player.players.get_mut(&handle)
        && player.video_buffers[0] == 0
    {
        player.video_buffers = buffers;
        player.video_buffer_stride = request.3;
        eprintln!(
            "[REMu][AVPLAYER] allocated guest video buffers size={} addresses={buffers:#x?}",
            request.3
        );
        true
    } else {
        false
    };
    drop(state);
    if !installed {
        // The player may have closed or another GetVideoData caller may have won the race while
        // the guest allocator callbacks ran. Return this losing allocation immediately.
        invoke_av_player_texture_deallocations(
            buffers
                .into_iter()
                .map(|address| AvPlayerTextureDeallocation {
                    function: request.1,
                    object: request.2,
                    address,
                })
                .collect(),
        );
    }
}

pub(crate) fn av_player_add_source(arguments: [u64; 6], state: &mut ProbeDispatchState) -> u64 {
    let Some(source) = read_guest_cstring(arguments[1]) else {
        return sce_kernel_error(AV_PLAYER_ERROR_INVALID_PARAMS);
    };
    let Some(host_path) = av_player_resolve_source(&state.game_root, &source) else {
        eprintln!("[REMu][AVPLAYER] rejected source outside app0: {source:?}");
        return sce_kernel_error(AV_PLAYER_ERROR_OPERATION_FAILED);
    };
    let Some((width, height, frames_per_second, duration)) = av_player_probe_video(&host_path)
    else {
        eprintln!(
            "[REMu][AVPLAYER] could not probe source guest={source:?} host={}",
            host_path.display()
        );
        return sce_kernel_error(AV_PLAYER_ERROR_OPERATION_FAILED);
    };
    let Some(player) = state.av_player.players.get_mut(&arguments[0]) else {
        return sce_kernel_error(AV_PLAYER_ERROR_INVALID_PARAMS);
    };
    player.reset_playback();
    player.source = Some(host_path.to_string_lossy().into_owned());
    player.width = width;
    player.height = height;
    player.frames_per_second = frames_per_second;
    player.duration = duration;
    player.has_audio = av_player_probe_has_audio(&host_path);
    player.active_session = true;
    player.paused = false;
    if player.auto_start {
        player.playing_since = Some(Instant::now());
    }
    let callback = (
        player.event_callback,
        player.event_object,
        player.auto_start,
    );
    eprintln!(
        "[REMu][AVPLAYER] source handle={:#x} guest={source:?} host={} {}x{} fps={frames_per_second:.3} duration_ms={} audio={} auto_start={}",
        arguments[0],
        host_path.display(),
        width,
        height,
        duration.as_millis(),
        player.has_audio,
        player.auto_start,
    );
    av_player_queue_event(
        callback.0,
        callback.1,
        2,
        AvPlayerEventDelivery::Controller,
        state,
    );
    if callback.2 {
        // Auto-start is performed as part of the controller's Ready transition.
        av_player_queue_event(
            callback.0,
            callback.1,
            3,
            AvPlayerEventDelivery::Controller,
            state,
        );
    }
    0
}

pub(crate) fn av_player_add_source_ex(arguments: [u64; 6], state: &mut ProbeDispatchState) -> u64 {
    let details = arguments[2];
    if arguments[1] != 0 || details == 0 {
        return sce_kernel_error(AV_PLAYER_ERROR_INVALID_PARAMS);
    }
    if !process_guest_range_has_read_access(details, 12) {
        return sce_kernel_error(AV_PLAYER_ERROR_INVALID_PARAMS);
    }
    let source_address = read_guest_u64(details);
    let length = u64::from(read_guest_u32(details + 8));
    let Ok(length) = usize::try_from(length) else {
        return sce_kernel_error(AV_PLAYER_ERROR_INVALID_PARAMS);
    };
    if source_address == 0 || length == 0 {
        return sce_kernel_error(AV_PLAYER_ERROR_INVALID_PARAMS);
    }
    if !process_guest_range_has_read_access(source_address, length) {
        return sce_kernel_error(AV_PLAYER_ERROR_INVALID_PARAMS);
    }
    // SAFETY: the source-details URI pairs this pointer with the exact readable byte length.
    let bytes = unsafe { std::slice::from_raw_parts(source_address as *const u8, length) };
    let Ok(source) = std::str::from_utf8(bytes) else {
        return sce_kernel_error(AV_PLAYER_ERROR_INVALID_PARAMS);
    };
    let terminated = format!("{}\0", source.trim_end_matches('\0'));
    let forwarded = [arguments[0], terminated.as_ptr() as u64, 0, 0, 0, 0];
    av_player_add_source(forwarded, state)
}

fn av_player_start_decoder(player: &mut AvPlayerInstance, audio: bool) -> bool {
    if if audio {
        player.audio_decoder.is_some()
    } else {
        player.video_decoder.is_some()
    } {
        return true;
    }
    let (Some(ffmpeg), Some(source)) = (
        av_player_tool("ffmpeg", "REMU_FFMPEG"),
        player.source.as_deref(),
    ) else {
        return false;
    };
    let mut command = Command::new(ffmpeg);
    command.args(["-nostdin", "-hide_banner", "-loglevel", "error"]);
    if !player.decoder_seek.is_zero() {
        // JumpToTime restarts the decoders; the input-side seek makes them begin at the target
        // instead of replaying the stream from byte zero.
        command.args(["-ss", &format!("{:.3}", player.decoder_seek.as_secs_f64())]);
    }
    command.arg("-i").arg(source);
    if audio {
        command.args([
            "-map", "0:a:0", "-vn", "-ac", "2", "-ar", "48000", "-f", "s16le", "pipe:1",
        ]);
    } else {
        command.args([
            "-map", "0:v:0", "-an", "-pix_fmt", "nv12", "-f", "rawvideo", "pipe:1",
        ]);
    }
    let frame_bytes = if audio {
        Some(1024 * 2 * 2)
    } else {
        u64::from(player.width)
            .checked_mul(u64::from(player.height))
            .and_then(|pixels| pixels.checked_mul(3))
            .and_then(|bytes| bytes.checked_div(2))
            .and_then(|bytes| usize::try_from(bytes).ok())
    };
    let Some(frame_bytes) = frame_bytes else {
        return false;
    };
    let Ok(mut child) = command.stdout(Stdio::piped()).stderr(Stdio::null()).spawn() else {
        return false;
    };
    let Some(output) = child.stdout.take() else {
        let _ = child.kill();
        let _ = child.wait();
        return false;
    };
    let (sender, frames) = mpsc::sync_channel(if audio { 8 } else { 4 });
    let cancelled = Arc::new(AtomicBool::new(false));
    let worker_cancelled = Arc::clone(&cancelled);
    let kind = if audio { "audio" } else { "video" };
    let Ok(worker) = std::thread::Builder::new()
        .name(format!("REMu AVPlayer {kind}"))
        .spawn(move || {
            av_player_decoder_worker(output, frame_bytes, sender, worker_cancelled);
        })
    else {
        let _ = child.kill();
        let _ = child.wait();
        return false;
    };
    let decoder = AvPlayerDecoder {
        child: Some(child),
        frames: std::sync::Mutex::new(frames),
        cancelled,
        worker: Some(worker),
    };
    if audio {
        player.raw_audio_frame.resize(frame_bytes, 0);
        player.audio_decoder = Some(decoder);
    } else {
        player.raw_video_frame.resize(frame_bytes, 0);
        player.video_decoder = Some(decoder);
    }
    player.playing_since.get_or_insert_with(Instant::now);
    eprintln!(
        "[REMu][AVPLAYER] started {} decoder source={source:?}",
        if audio { "audio" } else { "video" }
    );
    true
}

fn av_player_read_video_frame(player: &mut AvPlayerInstance) -> AvPlayerFrameStatus {
    let Some(decoder) = player.video_decoder.as_ref() else {
        return AvPlayerFrameStatus::Finished;
    };
    match decoder.poll() {
        AvPlayerDecoderPoll::Frame(frame) => {
            player.raw_video_frame = frame;
            AvPlayerFrameStatus::Ready
        }
        AvPlayerDecoderPoll::Pending => AvPlayerFrameStatus::Pending,
        AvPlayerDecoderPoll::Finished => AvPlayerFrameStatus::Finished,
    }
}

fn av_player_read_audio_frame(player: &mut AvPlayerInstance) -> AvPlayerFrameStatus {
    let Some(decoder) = player.audio_decoder.as_ref() else {
        return AvPlayerFrameStatus::Finished;
    };
    match decoder.poll() {
        AvPlayerDecoderPoll::Frame(frame) => {
            player.raw_audio_frame = frame;
            AvPlayerFrameStatus::Ready
        }
        AvPlayerDecoderPoll::Pending => AvPlayerFrameStatus::Pending,
        AvPlayerDecoderPoll::Finished => AvPlayerFrameStatus::Finished,
    }
}

fn av_player_finish_stream(player: &mut AvPlayerInstance) {
    if player.looping {
        player.reset_playback();
        player.playing_since = Some(Instant::now());
    } else {
        let started = player.playing_since.take();
        player.stop_decoders();
        player.active_session = false;
        player.end_of_stream = true;
        if let Some(started) = started {
            player.elapsed += started.elapsed();
        }
    }
}

pub(crate) fn av_player_nv12_layout(
    width: u32,
    height: u32,
    extended: bool,
) -> Option<(usize, usize)> {
    let width = usize::try_from(width).ok()?;
    let height = usize::try_from(height).ok()?;
    // The Gen5 extended ABI exposes a 256-byte-aligned pitch and the visible height. Legacy
    // frame info has no explicit pitch and therefore uses its 16-pixel-aligned dimensions.
    let (row_pitch, storage_height) = if extended {
        (width.checked_add(255)? & !255, height)
    } else {
        (width.checked_add(15)? & !15, height.checked_add(15)? & !15)
    };
    let frame_bytes = row_pitch
        .checked_mul(storage_height)?
        .checked_mul(3)?
        .checked_div(2)?;
    Some((row_pitch, frame_bytes))
}

fn av_player_ensure_video_buffers(player: &mut AvPlayerInstance, extended: bool) -> bool {
    if player.video_buffers[0] != 0 {
        return true;
    }
    let Some((_, stride)) = av_player_nv12_layout(player.width, player.height, extended) else {
        return false;
    };
    let Some(total) = stride
        .checked_mul(3)
        .and_then(|bytes| u64::try_from(bytes).ok())
    else {
        return false;
    };
    let allocation = guest_aligned_alloc(0x4000, total);
    if allocation == 0 {
        return false;
    }
    // SAFETY: the dedicated aligned allocation has exactly `total` writable bytes.
    unsafe {
        std::ptr::write_bytes(
            allocation as *mut u8,
            0,
            usize::try_from(total).unwrap_or(0),
        )
    };
    player.video_buffer_base = allocation;
    for (index, buffer) in player.video_buffers.iter_mut().enumerate() {
        *buffer = allocation
            + u64::try_from(index)
                .unwrap_or(0)
                .saturating_mul(u64::try_from(stride).unwrap_or(0));
    }
    player.video_buffer_stride = stride;
    eprintln!(
        "[REMu][AVPLAYER] allocated fallback video buffers size={stride} base={allocation:#x}"
    );
    true
}

fn av_player_write_video_frame(
    player: &mut AvPlayerInstance,
    output: u64,
    timestamp_milliseconds: u64,
    extended: bool,
) -> bool {
    if !av_player_ensure_video_buffers(player, extended) {
        return false;
    }
    let Some((row_pitch, frame_bytes)) =
        av_player_nv12_layout(player.width, player.height, extended)
    else {
        return false;
    };
    if player.video_buffer_stride < frame_bytes {
        return false;
    }
    let width = usize::try_from(player.width).unwrap_or(0);
    let height = usize::try_from(player.height).unwrap_or(0);
    let storage_height = if extended {
        height
    } else {
        height.saturating_add(15) & !15
    };
    let frame = if row_pitch == width {
        &player.raw_video_frame
    } else {
        player.padded_video_frame.resize(frame_bytes, 0);
        player.padded_video_frame.fill(0);
        for row in 0..height {
            let source = row * width;
            let destination = row * row_pitch;
            player.padded_video_frame[destination..destination + width]
                .copy_from_slice(&player.raw_video_frame[source..source + width]);
        }
        let source_chroma = width * height;
        let destination_chroma = row_pitch * storage_height;
        for row in 0..height.div_ceil(2) {
            let source = source_chroma + row * width;
            let destination = destination_chroma + row * row_pitch;
            player.padded_video_frame[destination..destination + width]
                .copy_from_slice(&player.raw_video_frame[source..source + width]);
        }
        &player.padded_video_frame
    };
    let address = player.video_buffers[player.next_video_buffer];
    player.next_video_buffer = (player.next_video_buffer + 1) % 3;
    // SAFETY: the destination is one complete slice of the dedicated three-frame allocation.
    unsafe { std::ptr::copy_nonoverlapping(frame.as_ptr(), address as *mut u8, frame.len()) };
    av_player_write_video_frame_info(
        output,
        address,
        timestamp_milliseconds,
        player.width,
        player.height,
        row_pitch,
        player.frames_per_second,
        extended,
    );
    true
}

pub(crate) fn av_player_write_video_frame_info(
    output: u64,
    address: u64,
    timestamp_milliseconds: u64,
    width: u32,
    height: u32,
    row_pitch: usize,
    frames_per_second: f64,
    extended: bool,
) {
    let visible_width = usize::try_from(width).unwrap_or(0);
    let visible_height = usize::try_from(height).unwrap_or(0);
    let output_width = if extended {
        row_pitch
    } else {
        visible_width.saturating_add(15) & !15
    };
    let output_height = if extended {
        visible_height
    } else {
        visible_height.saturating_add(15) & !15
    };
    guest_memset(output, 0, if extended { 104 } else { 40 });
    write_guest_u64(output, address);
    write_guest_u64(output + 16, timestamp_milliseconds);
    write_guest_u32(output + 24, u32::try_from(output_width).unwrap_or(width));
    write_guest_u32(output + 28, u32::try_from(output_height).unwrap_or(height));
    write_guest_u32(output + 32, 1.0_f32.to_bits());
    if extended {
        write_guest_u32(output + 60, u32::try_from(row_pitch).unwrap_or(width));
        write_guest_u8(output + 64, 8);
        write_guest_u8(output + 65, 8);
        // Silent Hill consumes the PS4-compatible 104-byte Gen5 VideoFrameInfoEx prefix. Crop
        // fields are left, right, top, bottom at +44/+48/+52/+56 respectively; writing padding
        // into left/top shifts the visible image even though its final extent happens to match.
        write_guest_u32(
            output + 48,
            u32::try_from(row_pitch.saturating_sub(visible_width)).unwrap_or(0),
        );
        write_guest_u32(
            output + 56,
            u32::try_from(output_height.saturating_sub(visible_height)).unwrap_or(0),
        );
        // Gen5 replaces the legacy integer frame-rate field with a double in the extended tail.
        write_guest_u64(output + 0x48, frames_per_second.to_bits());
    }
}

pub(crate) fn av_player_video_frame_is_due(
    next_video_frame: u64,
    frames_per_second: f64,
    media_clock: Duration,
) -> bool {
    let timestamp = Duration::from_secs_f64(next_video_frame as f64 / frames_per_second.max(1.0));
    timestamp <= media_clock
}

pub(crate) fn av_player_video_catch_up_target(
    origin: Option<(Instant, u64)>,
    next_video_frame: u64,
    frames_per_second: f64,
    duration: Duration,
    frame_bytes: usize,
    now: Instant,
) -> u64 {
    let Some((origin_time, origin_frame)) = origin else {
        // Always expose the first frame. Establishing the catch-up clock before this point made a
        // slow first emulated draw skip an entire short movie without showing any splash frame.
        return next_video_frame;
    };
    let elapsed_frames = (now.saturating_duration_since(origin_time).as_secs_f64()
        * frames_per_second.max(1.0))
    .floor() as u64;
    let mut target = origin_frame.saturating_add(elapsed_frames);
    if !duration.is_zero() {
        let total_frames = (duration.as_secs_f64() * frames_per_second.max(1.0)).ceil() as u64;
        target = target.min(total_frames.saturating_sub(1));
    }

    // Sequentially draining ffmpeg preserves decoder state and exact frame ordering. Bound the
    // work by bytes as well as frames so a tiny UI movie can catch up promptly without one poll
    // reading hundreds of MiB from a 4K stream.
    const MAXIMUM_CATCH_UP_BYTES: usize = 64 * 1_024 * 1_024;
    const MAXIMUM_CATCH_UP_FRAMES: u64 = 120;
    let byte_limited_frames = if frame_bytes == 0 {
        1
    } else {
        u64::try_from(MAXIMUM_CATCH_UP_BYTES / frame_bytes)
            .unwrap_or(MAXIMUM_CATCH_UP_FRAMES)
            .max(1)
    };
    let maximum_skip = byte_limited_frames.min(MAXIMUM_CATCH_UP_FRAMES);
    target.min(next_video_frame.saturating_add(maximum_skip))
}

fn av_player_wall_clock_catch_up_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| {
        std::env::var_os("REMU_AVPLAYER_WALL_CLOCK_CATCH_UP").is_some_and(|value| value == "1")
    })
}

fn av_player_trace_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED
        .get_or_init(|| std::env::var_os("REMU_TRACE_AVPLAYER").is_some_and(|value| value == "1"))
}

/// Counts every zero-return from the data exports by reason so a stalled guest player can be
/// diagnosed from one run: which gate starves it, and how often.
fn av_player_trace_zero_return(handle: u64, kind: &'static str, reason: &'static str) {
    use std::collections::HashMap;
    use std::sync::Mutex;
    if !av_player_trace_enabled() {
        return;
    }
    static COUNTS: OnceLock<Mutex<HashMap<(&'static str, &'static str), u64>>> = OnceLock::new();
    let mut counts = COUNTS
        .get_or_init(|| Mutex::new(HashMap::new()))
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let count = counts.entry((kind, reason)).or_default();
    *count += 1;
    if *count <= 8 || count.is_power_of_two() {
        eprintln!(
            "[REMu][AVPLAYER][TRACE] handle={handle:#x} {kind} zero-return reason={reason} count={count}"
        );
    }
}

/// Samples the public media clock without making high-frequency ticker calls dominate a trace.
///
/// In addition to the usual logarithmic call counts, report the exact call that first reaches the
/// known duration. That distinguishes a guest which stops polling because its clock completed
/// from one which stopped while the public clock was still progressing.
fn av_player_trace_current_time(handle: u64, player: &AvPlayerInstance, current: Duration) {
    use std::collections::HashMap;
    use std::sync::Mutex;

    if !av_player_trace_enabled() {
        return;
    }
    static STATES: OnceLock<Mutex<HashMap<u64, (u64, Duration)>>> = OnceLock::new();
    let mut states = STATES
        .get_or_init(|| Mutex::new(HashMap::new()))
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let (count, previous) = states.entry(handle).or_default();
    *count += 1;
    let first_at_duration =
        !player.duration.is_zero() && current == player.duration && *previous != current;
    if *count <= 8 || count.is_power_of_two() || first_at_duration {
        eprintln!(
            "[REMu][AVPLAYER][TRACE] handle={handle:#x} current-time count={count} value_ms={} raw_ms={} duration_ms={} active={} playing={} paused={} eos={} next_video_frame={} video_decoder={}",
            current.as_millis(),
            player.media_clock_elapsed().as_millis(),
            player.duration.as_millis(),
            player.active_session,
            player.playing_since.is_some(),
            player.paused,
            player.end_of_stream,
            player.next_video_frame,
            player.video_decoder.is_some(),
        );
    }
    *previous = current;
}

pub(crate) fn av_player_close(handle: u64, state: &mut ProbeDispatchState) -> u64 {
    let Some(mut player) = state.av_player.players.remove(&handle) else {
        return sce_kernel_error(AV_PLAYER_ERROR_INVALID_PARAMS);
    };
    if player.video_buffer_base == 0 && player.deallocate_texture_callback != 0 {
        state.av_player.pending_texture_deallocations.extend(
            player
                .video_buffers
                .into_iter()
                .filter(|address| *address != 0)
                .map(|address| AvPlayerTextureDeallocation {
                    function: player.deallocate_texture_callback,
                    object: player.allocator_object,
                    address,
                }),
        );
        // The callbacks now own these addresses. Clear the instance copy before its teardown so
        // future Drop changes cannot accidentally treat callback-owned memory as fallback memory.
        player.video_buffers = [0; 3];
    }
    0
}

pub(crate) fn av_player_current_time(handle: u64, state: &ProbeDispatchState) -> u64 {
    let Some(player) = state.av_player.players.get(&handle) else {
        return sce_kernel_error(AV_PLAYER_ERROR_INVALID_PARAMS);
    };
    // CurrentTime is the playback clock, not the timestamp of the last frame the guest happened
    // to dequeue. Video-only clients use this clock to decide when to poll GetVideoData; tying it
    // to delivered video creates a circular wait when the render-side poller is initially idle.
    // Bound a running clock by the known source duration, matching the native end-of-stream
    // contract while leaving sources with an unknown duration unbounded.
    let elapsed = player.media_clock_elapsed();
    let current = if player.duration.is_zero() {
        elapsed
    } else {
        elapsed.min(player.duration)
    };
    av_player_trace_current_time(handle, player, current);
    u64::try_from(current.as_millis()).unwrap_or(u64::MAX)
}

pub(crate) fn av_player_is_active(handle: u64, state: &ProbeDispatchState) -> u64 {
    state.av_player.players.get(&handle).map_or(0, |player| {
        // Source activity is independent of its media clock. In particular, AddSource's Ready
        // state and Pause both remain active; only Stop, Error/EOF, or Close ends the session.
        // Unity queries this Ready window before calling Start.
        u64::from(player.active_session && !player.end_of_stream)
    })
}

pub(crate) fn av_player_pause(handle: u64, state: &mut ProbeDispatchState) -> u64 {
    let Some(player) = state.av_player.players.get_mut(&handle) else {
        return sce_kernel_error(AV_PLAYER_ERROR_INVALID_PARAMS);
    };
    if player.end_of_stream {
        return 0;
    }
    if player.paused || player.playing_since.is_none() {
        return sce_kernel_error(AV_PLAYER_ERROR_OPERATION_FAILED);
    }
    if let Some(started) = player.playing_since.take() {
        player.elapsed += started.elapsed();
    }
    player.paused = true;
    player.video_catch_up_origin = None;
    let callback = (player.event_callback, player.event_object);
    av_player_queue_event(
        callback.0,
        callback.1,
        4,
        AvPlayerEventDelivery::CallingThread,
        state,
    );
    0
}

pub(crate) fn av_player_start(handle: u64, state: &mut ProbeDispatchState) -> u64 {
    let Some(player) = state.av_player.players.get_mut(&handle) else {
        return sce_kernel_error(AV_PLAYER_ERROR_INVALID_PARAMS);
    };
    if player.source.is_none() {
        return sce_kernel_error(AV_PLAYER_ERROR_OPERATION_FAILED);
    }
    player.paused = false;
    player.active_session = true;
    player.end_of_stream = false;
    player.video_catch_up_origin = None;
    player.playing_since.get_or_insert_with(Instant::now);
    let callback = (player.event_callback, player.event_object);
    eprintln!("[REMu][AVPLAYER] started handle={handle:#x}");
    av_player_queue_event(
        callback.0,
        callback.1,
        3,
        AvPlayerEventDelivery::CallingThread,
        state,
    );
    0
}

pub(crate) fn av_player_resume(handle: u64, state: &mut ProbeDispatchState) -> u64 {
    let Some(player) = state.av_player.players.get_mut(&handle) else {
        return sce_kernel_error(AV_PLAYER_ERROR_INVALID_PARAMS);
    };
    if !player.paused || player.source.is_none() {
        return sce_kernel_error(AV_PLAYER_ERROR_OPERATION_FAILED);
    }
    player.paused = false;
    player.active_session = true;
    player.video_catch_up_origin = None;
    player.playing_since = Some(Instant::now());
    let callback = (player.event_callback, player.event_object);
    eprintln!("[REMu][AVPLAYER] resumed handle={handle:#x}");
    av_player_queue_event(
        callback.0,
        callback.1,
        3,
        AvPlayerEventDelivery::CallingThread,
        state,
    );
    0
}

pub(crate) fn av_player_jump_to_time(arguments: [u64; 6], state: &mut ProbeDispatchState) -> u64 {
    let Some(player) = state.av_player.players.get_mut(&arguments[0]) else {
        return sce_kernel_error(AV_PLAYER_ERROR_INVALID_PARAMS);
    };
    // stop_decoders clears the running wall-clock segment, so capture it first: a jump during
    // playback must keep playing from the target instead of silently pausing.
    let was_playing = player.playing_since.is_some();
    player.stop_decoders();
    let target = Duration::from_millis(arguments[1]);
    // Freshly started decoders are launched with an input-side seek to the target; the frame
    // counters are aligned to the frames that seek actually emits, so delivered timestamps
    // continue to reflect stream position.
    player.decoder_seek = target;
    player.active_session = true;
    player.elapsed = target;
    player.end_of_stream = false;
    player.next_video_frame = (target.as_secs_f64() * player.frames_per_second).round() as u64;
    player.video_catch_up_origin = None;
    player.next_audio_frame = u64::try_from(target.as_millis())
        .unwrap_or(u64::MAX)
        .saturating_mul(48_000)
        / (1024 * 1_000);
    if was_playing {
        player.playing_since = Some(Instant::now());
    }
    eprintln!(
        "[REMu][AVPLAYER] jump handle={:#x} target_ms={} video_frame={} audio_frame={} playing={was_playing} paused={}",
        arguments[0], arguments[1], player.next_video_frame, player.next_audio_frame, player.paused,
    );
    0
}

pub(crate) fn av_player_stop(handle: u64, state: &mut ProbeDispatchState) -> u64 {
    let Some(player) = state.av_player.players.get_mut(&handle) else {
        return sce_kernel_error(AV_PLAYER_ERROR_INVALID_PARAMS);
    };
    player.reset_playback();
    player.active_session = false;
    player.paused = false;
    let callback = (player.event_callback, player.event_object);
    av_player_queue_event(
        callback.0,
        callback.1,
        1,
        AvPlayerEventDelivery::CallingThread,
        state,
    );
    0
}

pub(crate) fn av_player_set_looping(arguments: [u64; 6], state: &mut ProbeDispatchState) -> u64 {
    let Some(player) = state.av_player.players.get_mut(&arguments[0]) else {
        return sce_kernel_error(AV_PLAYER_ERROR_INVALID_PARAMS);
    };
    player.looping = arguments[1] & 1 != 0;
    0
}

/// sceAvPlayerChangeStream switches the active representation of a stream. Every source this
/// backend serves has exactly one representation per stream (one video, at most one audio), so
/// selecting any existing stream is already satisfied; anything else is a genuinely
/// unimplemented multi-representation switch and fails loudly with the observed state.
pub(crate) fn av_player_change_stream(arguments: [u64; 6], state: &ProbeDispatchState) -> u64 {
    let Some(player) = state.av_player.players.get(&arguments[0]) else {
        return sce_kernel_error(AV_PLAYER_ERROR_INVALID_PARAMS);
    };
    let stream_count = 1 + u64::from(player.has_audio);
    if arguments[1] < stream_count {
        eprintln!(
            "[REMu][AVPLAYER] change stream handle={:#x} stream={} already active (single representation)",
            arguments[0], arguments[1],
        );
        return 0;
    }
    eprintln!(
        "[REMu][AVPLAYER] ChangeStream unimplemented for handle={:#x}: requested stream={} arguments={:#x?} but source {:?} exposes {stream_count} single-representation streams",
        arguments[0],
        arguments[1],
        &arguments[1..3],
        player.source,
    );
    sce_kernel_error(AV_PLAYER_ERROR_INVALID_PARAMS)
}

/// sceAvPlayerSetTrickSpeed sets the playback rate in percent (100 = normal). The rate scales
/// the player media clock; media timestamps are unchanged. Reverse and paused-rate trick
/// play would need a seeking decoder, so non-positive rates fail loudly instead of guessing.
pub(crate) fn av_player_set_trick_speed(
    arguments: [u64; 6],
    state: &mut ProbeDispatchState,
) -> u64 {
    let Some(player) = state.av_player.players.get_mut(&arguments[0]) else {
        return sce_kernel_error(AV_PLAYER_ERROR_INVALID_PARAMS);
    };
    #[allow(clippy::cast_possible_truncation)]
    let speed = arguments[1] as i32;
    let Ok(speed) = u32::try_from(speed) else {
        eprintln!(
            "[REMu][AVPLAYER] SetTrickSpeed unimplemented for handle={:#x}: reverse rate {speed}% needs a seeking decoder",
            arguments[0],
        );
        return sce_kernel_error(AV_PLAYER_ERROR_OPERATION_FAILED);
    };
    if speed == 0 || speed > 3_200 {
        eprintln!(
            "[REMu][AVPLAYER] SetTrickSpeed rejected handle={:#x} rate={speed}%: outside the 1..=3200 forward range",
            arguments[0],
        );
        return sce_kernel_error(AV_PLAYER_ERROR_INVALID_PARAMS);
    }
    player.trick_speed_percent = speed;
    eprintln!(
        "[REMu][AVPLAYER] trick speed handle={:#x} rate={speed}%",
        arguments[0],
    );
    0
}

pub(crate) fn av_player_get_video_data(
    arguments: [u64; 6],
    extended: bool,
    state: &mut ProbeDispatchState,
) -> u64 {
    let Some(player) = state.av_player.players.get_mut(&arguments[0]) else {
        return 0;
    };
    if arguments[1] == 0 {
        av_player_trace_zero_return(arguments[0], "video", "null-output");
        return 0;
    }
    // Match the native state machine: decoder threads may continue filling their private queues,
    // but neither data export releases a frame while the player is paused.
    if player.paused {
        av_player_trace_zero_return(arguments[0], "video", "paused");
        return 0;
    }
    if player.playing_since.is_none() {
        av_player_trace_zero_return(arguments[0], "video", "stopped");
        return 0;
    }
    if player.end_of_stream {
        av_player_trace_zero_return(arguments[0], "video", "end-of-stream");
        return 0;
    }
    if !av_player_start_decoder(player, false) {
        av_player_trace_zero_return(arguments[0], "video", "no-decoder");
        return 0;
    }
    if !av_player_video_frame_is_due(
        player.next_video_frame,
        player.frames_per_second,
        player.media_clock_elapsed(),
    ) {
        av_player_trace_zero_return(arguments[0], "video", "clock-not-reached");
        return 0;
    }
    // Completed VideoOut flips are the emulated media clock. Wall-clock catch-up is useful only
    // when translation already sustains realtime; below that rate it discarded nearly every
    // Silent Hill splash/movie frame while audio and guest time continued ahead. Keep it as an
    // explicit diagnostic option rather than the correctness default.
    let catch_up_target = if av_player_wall_clock_catch_up_enabled() && !player.paused {
        av_player_video_catch_up_target(
            player.video_catch_up_origin,
            player.next_video_frame,
            player.frames_per_second,
            player.duration,
            player.raw_video_frame.len(),
            Instant::now(),
        )
    } else {
        player.next_video_frame
    };
    let skipped_frames = catch_up_target.saturating_sub(player.next_video_frame);
    for _ in 0..skipped_frames {
        match av_player_read_video_frame(player) {
            AvPlayerFrameStatus::Ready => {}
            AvPlayerFrameStatus::Pending => {
                av_player_trace_zero_return(arguments[0], "video", "decoder-pending");
                return 0;
            }
            AvPlayerFrameStatus::Finished => {
                av_player_finish_stream(player);
                return 0;
            }
        }
        player.next_video_frame = player.next_video_frame.saturating_add(1);
    }
    if skipped_frames != 0 {
        eprintln!(
            "[REMu][AVPLAYER] caught up {skipped_frames} frames to frame={} wall_ms={}",
            player.next_video_frame,
            player.playback_elapsed().as_millis()
        );
    }
    match av_player_read_video_frame(player) {
        AvPlayerFrameStatus::Ready => {}
        AvPlayerFrameStatus::Pending => {
            av_player_trace_zero_return(arguments[0], "video", "decoder-pending");
            return 0;
        }
        AvPlayerFrameStatus::Finished => {
            av_player_finish_stream(player);
            return 0;
        }
    }
    let timestamp = ((player.next_video_frame as f64) * 1_000.0 / player.frames_per_second.max(1.0))
        .round() as u64;
    player.next_video_frame = player.next_video_frame.saturating_add(1);
    if av_player_write_video_frame(player, arguments[1], timestamp, extended) {
        player
            .video_catch_up_origin
            .get_or_insert((Instant::now(), player.next_video_frame));
        if player.next_video_frame <= 4 || player.next_video_frame.is_power_of_two() {
            eprintln!(
                "[REMu][AVPLAYER] video frame={} timestamp_ms={timestamp} extended={extended}",
                player.next_video_frame
            );
        }
        1
    } else {
        0
    }
}

pub(crate) fn av_player_get_audio_data(arguments: [u64; 6], state: &mut ProbeDispatchState) -> u64 {
    const SAMPLES_PER_FRAME: usize = 1024;
    const CHANNELS: usize = 2;
    const SAMPLE_RATE: u32 = 48_000;
    const FRAME_BYTES: usize = SAMPLES_PER_FRAME * CHANNELS * 2;
    let Some(player) = state.av_player.players.get_mut(&arguments[0]) else {
        return 0;
    };
    if arguments[1] == 0 || !player.has_audio {
        av_player_trace_zero_return(arguments[0], "audio", "no-output-or-no-track");
        return 0;
    }
    if player.paused {
        av_player_trace_zero_return(arguments[0], "audio", "paused");
        return 0;
    }
    if player.playing_since.is_none() {
        av_player_trace_zero_return(arguments[0], "audio", "stopped");
        return 0;
    }
    if player.end_of_stream {
        av_player_trace_zero_return(arguments[0], "audio", "end-of-stream");
        return 0;
    }
    if !av_player_start_decoder(player, true) {
        av_player_trace_zero_return(arguments[0], "audio", "no-decoder");
        return 0;
    }
    match av_player_read_audio_frame(player) {
        AvPlayerFrameStatus::Ready => {}
        AvPlayerFrameStatus::Pending => {
            av_player_trace_zero_return(arguments[0], "audio", "decoder-pending");
            return 0;
        }
        AvPlayerFrameStatus::Finished => {
            av_player_trace_zero_return(arguments[0], "audio", "decoder-drained");
            return 0;
        }
    }
    if player.audio_buffer_base == 0 {
        let total = u64::try_from(FRAME_BYTES * 8).unwrap_or(0);
        player.audio_buffer_base = guest_aligned_alloc(0x100, total);
        if player.audio_buffer_base == 0 {
            return 0;
        }
    }
    let address = player.audio_buffer_base
        + u64::try_from(player.next_audio_buffer * FRAME_BYTES).unwrap_or(0);
    player.next_audio_buffer = (player.next_audio_buffer + 1) % 8;
    // SAFETY: `address` selects one complete frame in the dedicated eight-frame allocation.
    unsafe {
        std::ptr::copy_nonoverlapping(
            player.raw_audio_frame.as_ptr(),
            address as *mut u8,
            FRAME_BYTES,
        )
    };
    let timestamp = player
        .next_audio_frame
        .saturating_mul(SAMPLES_PER_FRAME as u64)
        * 1_000
        / u64::from(SAMPLE_RATE);
    player.next_audio_frame = player.next_audio_frame.saturating_add(1);
    guest_memset(arguments[1], 0, 40);
    write_guest_u64(arguments[1], address);
    write_guest_u64(arguments[1] + 16, timestamp);
    write_guest_u16(arguments[1] + 24, CHANNELS as u16);
    write_guest_u32(arguments[1] + 28, SAMPLE_RATE);
    write_guest_u32(arguments[1] + 32, FRAME_BYTES as u32);
    1
}

pub(crate) fn av_player_stream_count(handle: u64, state: &ProbeDispatchState) -> u64 {
    state.av_player.players.get(&handle).map_or_else(
        || sce_kernel_error(AV_PLAYER_ERROR_INVALID_PARAMS),
        |player| 1 + u64::from(player.has_audio),
    )
}

pub(crate) fn av_player_validate_handle(handle: u64, state: &ProbeDispatchState) -> u64 {
    if state.av_player.players.contains_key(&handle) {
        0
    } else {
        sce_kernel_error(AV_PLAYER_ERROR_INVALID_PARAMS)
    }
}

// Gen5 callers use one-based stream discriminators: video=1, audio=2, timed text=3. This differs
// from the zero-based PS4 ABI. Both the local Unity and Unreal call sites compare the legacy
// stream-info type against these values before selecting GetVideoData or GetAudioData.
pub(crate) const AV_PLAYER_STREAM_TYPE_VIDEO: u32 = 1;
pub(crate) const AV_PLAYER_STREAM_TYPE_AUDIO: u32 = 2;
const AV_PLAYER_STREAM_INFO_SIZE: u64 = 32;
const AV_PLAYER_STREAM_INFO_EX_SIZE: u64 = 104;

pub(crate) fn av_player_write_stream_info(
    output: u64,
    stream_index: u64,
    width: u32,
    height: u32,
    duration: Duration,
) {
    guest_memset(output, 0, AV_PLAYER_STREAM_INFO_SIZE);
    let stream_type = if stream_index == 0 {
        AV_PLAYER_STREAM_TYPE_VIDEO
    } else {
        AV_PLAYER_STREAM_TYPE_AUDIO
    };
    write_guest_u32(output, stream_type);
    if stream_type == AV_PLAYER_STREAM_TYPE_VIDEO {
        write_guest_u32(output + 8, width.saturating_add(15) & !15);
        write_guest_u32(output + 12, height.saturating_add(15) & !15);
        write_guest_u32(output + 16, ((width as f32) / (height as f32)).to_bits());
    } else {
        write_guest_u16(output + 8, 2);
        write_guest_u32(output + 12, 48_000);
    }
    write_guest_u64(
        output + 24,
        u64::try_from(duration.as_millis()).unwrap_or(u64::MAX),
    );
}

pub(crate) fn av_player_write_stream_info_ex(
    output: u64,
    stream_index: u64,
    width: u32,
    height: u32,
    frames_per_second: f64,
    duration: Duration,
) {
    // Preserve the caller-populated this_size field. The remaining 96 bytes are the Gen5
    // discriminator, stream-specific payload, and duration.
    guest_memset(output + 8, 0, AV_PLAYER_STREAM_INFO_EX_SIZE - 8);
    let stream_type = if stream_index == 0 {
        AV_PLAYER_STREAM_TYPE_VIDEO
    } else {
        AV_PLAYER_STREAM_TYPE_AUDIO
    };
    write_guest_u32(output + 8, stream_type);
    if stream_type == AV_PLAYER_STREAM_TYPE_VIDEO {
        write_guest_u32(output + 16, width);
        write_guest_u32(output + 20, height);
        write_guest_u32(output + 24, ((width as f32) / (height as f32)).to_bits());
        write_guest_u64(output + 64, frames_per_second.to_bits());
    } else {
        write_guest_u16(output + 16, 2);
        write_guest_u32(output + 20, 48_000);
    }
    write_guest_u64(
        output + 96,
        u64::try_from(duration.as_millis()).unwrap_or(u64::MAX),
    );
}

pub(crate) fn av_player_get_stream_info(
    arguments: [u64; 6],
    extended: bool,
    state: &ProbeDispatchState,
) -> u64 {
    let Some(player) = state.av_player.players.get(&arguments[0]) else {
        return sce_kernel_error(AV_PLAYER_ERROR_INVALID_PARAMS);
    };
    if arguments[1] > u64::from(player.has_audio)
        || arguments[2] == 0
        || player.width == 0
        || player.height == 0
    {
        return sce_kernel_error(AV_PLAYER_ERROR_INVALID_PARAMS);
    }
    if extended {
        if read_guest_u64(arguments[2]) < AV_PLAYER_STREAM_INFO_EX_SIZE {
            return sce_kernel_error(AV_PLAYER_ERROR_INVALID_PARAMS);
        }
        av_player_write_stream_info_ex(
            arguments[2],
            arguments[1],
            player.width,
            player.height,
            player.frames_per_second,
            player.duration,
        );
    } else {
        // Gen5's legacy stream-info structure is 32 bytes and omits the PS4 start-time tail.
        av_player_write_stream_info(
            arguments[2],
            arguments[1],
            player.width,
            player.height,
            player.duration,
        );
    }
    eprintln!(
        "[REMu][AVPLAYER] stream info{} handle={:#x} index={} type={} {}x{} fps={:.3} duration_ms={}",
        if extended { " ex" } else { "" },
        arguments[0],
        arguments[1],
        if arguments[1] == 0 { "video" } else { "audio" },
        player.width,
        player.height,
        player.frames_per_second,
        player.duration.as_millis(),
    );
    0
}
