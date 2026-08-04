use std::collections::BTreeSet;
use std::sync::{Mutex, OnceLock};
use std::time::{Duration, Instant};

use crate::{
    AUDIO_OUT, AUDIO_OUT2, AUDIO_OUT2_CONTEXT_MEMORY_BASE_SIZE,
    AUDIO_OUT2_CONTEXT_MEMORY_QUEUE_STRIDE, AUDIO_OUT2_CONTEXT_PARAM_SIZE,
    AUDIO_OUT2_DEFAULT_GRAIN_FRAMES, AUDIO_OUT2_DEFAULT_QUEUE_DEPTH, AUDIO_OUT2_MAX_QUEUE_DEPTH,
    AUDIO_OUT2_PORT_STATE_SIZE, AUDIO_OUT2_PORT_TYPE_VIBRATION, AUDIO_OUT2_SPEAKER_INFO_SIZE,
    AudioOut2ContextState, AudioOut2PortRuntimeState, AudioOut2State,
    SCE_AUDIO_OUT2_ERROR_NOT_READY, SCE_KERNEL_ERROR_INVALID_ARGUMENT, SceAudioOut2Attribute,
    SceAudioOut2ContextParam, SceAudioOut2Pcm, SceAudioOut2PortParam, SceAudioOut2PortState, audio,
    audio_out_error, audio_out2_queue_wait_duration, audio_out2_update_context_queue,
    decode_audio_out2_attribute, decode_audio_out2_channel_format, guest_memset,
    guest_output_pointer, process_guest_range_has_read_access, process_guest_range_is_readable,
    read_audio_out2_context_param, read_audio_out2_port_param, read_guest_bytes, read_guest_u32,
    read_guest_u64, sce_kernel_error, trace_audio_out2_context_memory,
    trace_audio_out2_mastering_memory, trace_audio_out2_memory, trace_audio_out2_pcm,
    trace_logarithmic_sequence, write_audio_out2_context_param, write_audio_out2_port_state,
    write_guest_u16, write_guest_u32, write_guest_u64,
};

/// Validates an audio-output argument that REmu treats as a writable guest pointer.
///
/// These queries are documented only by their names, so an argument position can be a handle or a
/// selector rather than the output buffer. Writing through a small integer corrupts whatever the
/// host has mapped there, so a value that cannot be a guest address is refused and reported once
/// with the whole argument list, which is what identifies the real signature.
fn audio_out2_output_pointer(
    nid: &str,
    arguments: [u64; 6],
    candidate: u64,
    size: u64,
) -> Option<u64> {
    let Ok(size) = usize::try_from(size) else {
        return None;
    };
    if process_guest_range_is_readable(candidate, size) {
        return Some(candidate);
    }
    static REPORTED: OnceLock<Mutex<BTreeSet<String>>> = OnceLock::new();
    let first = REPORTED
        .get_or_init(|| Mutex::new(BTreeSet::new()))
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .insert(nid.to_owned());
    if first {
        eprintln!(
            "[REMu][AUDIO2] {nid} output {candidate:#x} is not a writable guest address; args={:#x},{:#x},{:#x},{:#x},{:#x},{:#x}",
            arguments[0], arguments[1], arguments[2], arguments[3], arguments[4], arguments[5]
        );
    }
    None
}

/// Rejects a PCM block after `ContextPush` reserved one virtual hardware-queue slot.
///
/// The reservation and the guest-memory snapshot form one operation. Returning an error without
/// undoing the reservation makes an invalid or concurrently unmapped PCM pointer consume queue
/// capacity until the device clock happens to drain it, so later valid nonblocking pushes can
/// spuriously report `NOT_READY`.
fn reject_audio_out2_reserved_pcm(
    state: &mut AudioOut2State,
    context_handle: u64,
    push: u64,
    port_handle: u64,
    address: u64,
    byte_length: Option<usize>,
    reason: &str,
) -> u64 {
    if let Some(context) = state.contexts.get_mut(&context_handle) {
        context.queued = context.queued.saturating_sub(1);
    }
    state.failed_context_pushes = state.failed_context_pushes.saturating_add(1);
    let failure = state.failed_context_pushes;
    if failure <= 8 || failure.is_power_of_two() {
        let byte_length = byte_length.map_or_else(
            || "overflow".to_owned(),
            |byte_length| byte_length.to_string(),
        );
        eprintln!(
            "[REMu][AUDIO2] context push={push} failure={failure} rejected port={port_handle} address={address:#x} bytes={byte_length} reason={reason}; released reserved queue slot"
        );
    }
    sce_kernel_error(SCE_KERNEL_ERROR_INVALID_ARGUMENT)
}

pub(crate) fn dispatch_audio_out2_import(nid: &str, arguments: [u64; 6]) -> u64 {
    let state = AUDIO_OUT2.get_or_init(|| Mutex::new(AudioOut2State::default()));
    let mut state = state
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    match nid {
        "g2tViFIohHE" => {
            eprintln!("[REMu][AUDIO2] initialized parameter={:#x}", arguments[0]);
            0
        }
        // Audio middleware treats mastering as a required second initialization stage. The
        // mastering graph itself does not change the interleaved blocks REmu sends to the host,
        // but rejecting this call makes the middleware tear down and recreate every user and
        // port without ever pushing a block.
        "XHl38ZNknbs" => {
            eprintln!(
                "[REMu][AUDIO2] initialized mastering config={:#x} state={:#x}",
                arguments[2], arguments[5]
            );
            0
        }
        // The host mixer consumes the already-mastered interleaved result. Accept graph parameter
        // updates so middleware can finish constructing that graph instead of entering its
        // teardown-and-retry path.
        "v8iOE+j8a5o" => {
            // Wwise embeds this parameter block 0x5b8 bytes into its mastering device object.
            // Retaining its address lets the opt-in memory diagnostic follow the large float
            // buffers owned by that object at the same instants ContextPush submits a grain.
            if arguments[0] != 0 && process_guest_range_is_readable(arguments[0], size_of::<u64>())
            {
                state.mastering_parameters = arguments[0];
            }
            0
        }
        // The ABI has two optional u32 outputs. Initial setup supplies both, while Wwise's live
        // device poll supplies only the second argument and deliberately leaves the third null.
        // Treating that valid poll as an error makes Wwise report a lost output device and rebuild
        // its entire graph on every watchdog interval.
        "R7d0F1g2qsU" => {
            let (queued, available, queue_depth, poll_backoff) = {
                let now = Instant::now();
                let Some(context) = state.contexts.get_mut(&arguments[0]) else {
                    return sce_kernel_error(SCE_KERNEL_ERROR_INVALID_ARGUMENT);
                };
                audio_out2_update_context_queue(context, now);
                let available = context.queue_depth.saturating_sub(context.queued);
                // Wwise polls only the available-slot output in a tight loop while the hardware
                // queue is full. A real syscall and device query naturally yield enough CPU for
                // the mixer thread to submit the next grain; an in-process HLE call can otherwise
                // execute tens of millions of polls per second and contend on this state mutex.
                // A bounded host scheduling backoff preserves the reported zero level while
                // releasing the mutex and keeps the polling thread from starving the producer.
                let poll_backoff =
                    (arguments[1] == 0 && arguments[2] != 0 && available == 0).then(|| {
                        audio_out2_queue_wait_duration(context, now).min(Duration::from_millis(1))
                    });
                (context.queued, available, context.queue_depth, poll_backoff)
            };
            let mut result = 0;
            for (level, value) in [(arguments[1], queued), (arguments[2], available)] {
                if level == 0 {
                    continue;
                }
                let Some(level) = guest_output_pointer(level, size_of::<u32>()) else {
                    result = sce_kernel_error(SCE_KERNEL_ERROR_INVALID_ARGUMENT);
                    break;
                };
                write_guest_u32(level, value);
            }
            state.queue_level_queries = state.queue_level_queries.saturating_add(1);
            let query = state.queue_level_queries;
            if query <= 64 || query.is_power_of_two() {
                eprintln!(
                    "[REMu][AUDIO2] queue level query={query} context={} queued={queued}/{}, available={available} outputs={:#x},{:#x} result={result:#x} args={:#x},{:#x},{:#x}",
                    arguments[0],
                    queue_depth,
                    arguments[1],
                    arguments[2],
                    arguments[3],
                    arguments[4],
                    arguments[5],
                );
            }
            drop(state);
            if let Some(backoff) = poll_backoff
                && !backoff.is_zero()
            {
                std::thread::sleep(backoff);
            }
            result
        }
        "gatEUKG+Ea4" => {
            let Some(output) =
                audio_out2_output_pointer(nid, arguments, arguments[1], AUDIO_OUT2_PORT_STATE_SIZE)
            else {
                return sce_kernel_error(SCE_KERNEL_ERROR_INVALID_ARGUMENT);
            };
            let port_exists = state.ports.contains_key(&arguments[0]);
            let port = state.ports.get(&arguments[0]).copied();
            let parameter = port.map(|port| port.parameter);
            let channel_count = port.map_or(2, |port| port.channel_count);
            state.port_state_queries = state.port_state_queries.saturating_add(1);
            let query = state.port_state_queries;
            // AudioOut2 exposes the same leading live-device state as the ordinary AudioOut
            // path. Byte 2 is the physical channel count, bytes 4..6 are the signed volume, and
            // bytes 6..8 are the reroute counter. Writing a volume byte at offset 7 fabricates a
            // large reroute count and makes middleware destroy and recreate every port.
            // Route 1 is the connected primary device, matching ordinary AudioOut. Route 0 means
            // disconnected: Wwise's device poll treats it as a removal and immediately rebuilds
            // every user and port, even though PortCreate and mastering both succeeded.
            write_audio_out2_port_state(
                output,
                SceAudioOut2PortState {
                    output: 1,
                    num_channels: channel_count,
                    volume: 127,
                    ..SceAudioOut2PortState::default()
                },
            );
            if query <= 8 || query.is_power_of_two() {
                eprintln!(
                    "[REMu][AUDIO2] port state query={query} handle={} exists={port_exists} output={output:#x} parameter={parameter:?} args={:#x},{:#x},{:#x},{:#x}",
                    arguments[0], arguments[2], arguments[3], arguments[4], arguments[5]
                );
            }
            0
        }
        // The speaker description is written through the first argument; the second is a flags
        // selector. Report one stable stereo device with its complete AudioOut2 topology.
        "DImz2Ft9E2g" => {
            let Some(info) = audio_out2_output_pointer(
                nid,
                arguments,
                arguments[0],
                AUDIO_OUT2_SPEAKER_INFO_SIZE,
            ) else {
                return sce_kernel_error(SCE_KERNEL_ERROR_INVALID_ARGUMENT);
            };
            // Layout: u8 type, three reserved bytes, u32 available-speaker mask, u32 flags,
            // one reserved u32, then twelve i16 azimuth/elevation pairs. Flags at offset 8 are
            // output state, not a connected bit. The stereo mask and conventional ±30-degree
            // positions describe the two available speakers.
            guest_memset(info, 0, AUDIO_OUT2_SPEAKER_INFO_SIZE);
            write_guest_u32(info + 4, 0x3);
            write_guest_u16(info + 0x10, (-30_i16).cast_unsigned());
            write_guest_u16(info + 0x14, 30);
            state.speaker_queries = state.speaker_queries.saturating_add(1);
            let query = state.speaker_queries;
            if query <= 8 || query.is_power_of_two() {
                eprintln!(
                    "[REMu][AUDIO2] speaker query={query} output={info:#x} args={:#x},{:#x},{:#x},{:#x},{:#x}",
                    arguments[1], arguments[2], arguments[3], arguments[4], arguments[5],
                );
            }
            0
        }
        "t5YrizufpQc" => {
            let Some(parameter) = guest_output_pointer(
                arguments[0],
                usize::try_from(AUDIO_OUT2_CONTEXT_PARAM_SIZE)
                    .expect("AudioOut2 parameter size fits usize"),
            ) else {
                return sce_kernel_error(SCE_KERNEL_ERROR_INVALID_ARGUMENT);
            };
            // SceAudioOut2ContextParam is a fixed 0x40-byte object. Reset supplies the hardware
            // defaults middleware retains unless it needs a different port count or grain size.
            write_audio_out2_context_param(
                parameter,
                SceAudioOut2ContextParam {
                    max_ports: 256,
                    max_object_ports: 256,
                    queue_depth: AUDIO_OUT2_DEFAULT_QUEUE_DEPTH,
                    num_grains: AUDIO_OUT2_DEFAULT_GRAIN_FRAMES,
                    flags: 1,
                    ..SceAudioOut2ContextParam::default()
                },
            );
            0
        }
        "pDmme7Bgm6E" => {
            if arguments[0] == 0 || arguments[1] == 0 {
                return sce_kernel_error(SCE_KERNEL_ERROR_INVALID_ARGUMENT);
            }
            let Some(parameter) = guest_output_pointer(
                arguments[0],
                usize::try_from(AUDIO_OUT2_CONTEXT_PARAM_SIZE)
                    .expect("AudioOut2 parameter size fits usize"),
            ) else {
                return sce_kernel_error(SCE_KERNEL_ERROR_INVALID_ARGUMENT);
            };
            let Some(memory_size) = guest_output_pointer(arguments[1], size_of::<u64>()) else {
                return sce_kernel_error(SCE_KERNEL_ERROR_INVALID_ARGUMENT);
            };
            // KytyPS5 derives this from the hardware queue depth. This reproduces the 0x11640
            // Wwise request with the default four queues without hard-coding that one setup.
            let parameter = read_audio_out2_context_param(parameter);
            let queue_depth = match parameter.queue_depth {
                0 => AUDIO_OUT2_DEFAULT_QUEUE_DEPTH,
                value => value,
            };
            let context_memory_size = AUDIO_OUT2_CONTEXT_MEMORY_BASE_SIZE.saturating_add(
                u64::from(queue_depth).saturating_mul(AUDIO_OUT2_CONTEXT_MEMORY_QUEUE_STRIDE),
            );
            write_guest_u64(memory_size, context_memory_size);
            eprintln!(
                "[REMu][AUDIO2] context memory query parameter={:#x} queue_depth={queue_depth} size={context_memory_size:#x}",
                arguments[0],
            );
            0
        }
        "0x6o1VVAYSY" => {
            if arguments[0] == 0 || arguments[1] == 0 || arguments[2] == 0 || arguments[3] == 0 {
                return sce_kernel_error(SCE_KERNEL_ERROR_INVALID_ARGUMENT);
            }
            let Some(parameter) = guest_output_pointer(
                arguments[0],
                usize::try_from(AUDIO_OUT2_CONTEXT_PARAM_SIZE)
                    .expect("AudioOut2 parameter size fits usize"),
            ) else {
                return sce_kernel_error(SCE_KERNEL_ERROR_INVALID_ARGUMENT);
            };
            let parameter = read_audio_out2_context_param(parameter);
            let queue_depth = if parameter.queue_depth == 0 {
                AUDIO_OUT2_DEFAULT_QUEUE_DEPTH
            } else {
                parameter.queue_depth
            };
            let grain_frames = if parameter.num_grains == 0 {
                AUDIO_OUT2_DEFAULT_GRAIN_FRAMES
            } else {
                parameter.num_grains
            };
            if queue_depth > AUDIO_OUT2_MAX_QUEUE_DEPTH
                || !(64..=8192).contains(&grain_frames)
                || !grain_frames.is_power_of_two()
            {
                eprintln!(
                    "[REMu][AUDIO2] rejected context parameter={parameter:?} queue_depth={queue_depth} grain={grain_frames}"
                );
                return sce_kernel_error(SCE_KERNEL_ERROR_INVALID_ARGUMENT);
            }
            let Ok(memory_size) = usize::try_from(arguments[2]) else {
                return sce_kernel_error(SCE_KERNEL_ERROR_INVALID_ARGUMENT);
            };
            if !process_guest_range_is_readable(arguments[1], memory_size) {
                eprintln!(
                    "[REMu][AUDIO2] rejected context memory={:#x}+{:#x}",
                    arguments[1], arguments[2]
                );
                return sce_kernel_error(SCE_KERNEL_ERROR_INVALID_ARGUMENT);
            }
            let handle = state.allocate_handle();
            state.contexts.insert(
                handle,
                AudioOut2ContextState {
                    queue_depth,
                    grain_frames,
                    queued: 0,
                    last_queue_update: Instant::now(),
                    last_device_frames: crate::audio::device_clock_frames().unwrap_or(0),
                    memory_address: arguments[1],
                    memory_size: arguments[2],
                },
            );
            write_guest_u64(arguments[3], handle);
            eprintln!(
                "[REMu][AUDIO2] created context={handle} memory={:#x} size={:#x} queue_depth={queue_depth} grain={grain_frames} parameter={parameter:?}",
                arguments[1], arguments[2],
            );
            0
        }
        "xywYcRB7nbQ" => {
            if arguments[1] == 0 {
                return sce_kernel_error(SCE_KERNEL_ERROR_INVALID_ARGUMENT);
            }
            let handle = state.allocate_handle();
            state.users.insert(handle);
            state.user_creates = state.user_creates.saturating_add(1);
            let create = state.user_creates;
            write_guest_u64(arguments[1], handle);
            if create <= 8 || create.is_power_of_two() {
                eprintln!(
                    "[REMu][AUDIO2] user create={create} handle={handle} user_id={:#x}",
                    arguments[0]
                );
            }
            0
        }
        "JK2wamZPzwM" => {
            let Some(context) = state.contexts.get(&arguments[0]).copied() else {
                return sce_kernel_error(SCE_KERNEL_ERROR_INVALID_ARGUMENT);
            };
            if arguments[1] == 0 || arguments[2] == 0 {
                return sce_kernel_error(SCE_KERNEL_ERROR_INVALID_ARGUMENT);
            }
            if !process_guest_range_is_readable(
                arguments[1],
                std::mem::size_of::<SceAudioOut2PortParam>(),
            ) {
                return sce_kernel_error(SCE_KERNEL_ERROR_INVALID_ARGUMENT);
            }
            let parameter = read_audio_out2_port_param(arguments[1]);
            let frequency = u64::from(parameter.sampling_frequency);
            let channel_format = parameter.data_format;
            let Some((channel_count, raw_format)) =
                decode_audio_out2_channel_format(channel_format)
            else {
                eprintln!(
                    "[REMu][AUDIO2] unsupported port channel format={channel_format:#x} parameter={parameter:?}"
                );
                return sce_kernel_error(SCE_KERNEL_ERROR_INVALID_ARGUMENT);
            };
            // Port type 6 carries controller-vibration samples. Sending those floats to the
            // ordinary speaker mixer produces loud garbage and is observably the wrong route.
            // Pad/personal/aux routing remains on the existing fallback until a distinct host
            // endpoint is implemented; only the known non-audible route is suppressed here.
            let backend_handle =
                (parameter.port_type != AUDIO_OUT2_PORT_TYPE_VIBRATION).then(|| {
                    let mut audio = AUDIO_OUT
                        .get_or_init(|| Mutex::new(audio::AudioOutState::default()))
                        .lock()
                        .unwrap_or_else(std::sync::PoisonError::into_inner);
                    audio.initialize();
                    // An AudioOut2 port is a mixer voice, not one of the eight hardware outputs
                    // `sceAudioOutOpen` rations, so it is opened against the voice budget.
                    match audio.open_within(
                        [
                            0,
                            0,
                            0,
                            u64::from(context.grain_frames),
                            frequency,
                            raw_format,
                        ],
                        audio::AudioOutState::MIXER_VOICE_BUDGET,
                    ) {
                        Ok(handle) => Ok(handle),
                        Err(error) => Err(audio_out_error(error)),
                    }
                });
            let backend_handle = match backend_handle.transpose() {
                Ok(handle) => handle,
                Err(error) => return error,
            };
            let handle = state.allocate_handle();
            state.ports.insert(
                handle,
                AudioOut2PortRuntimeState {
                    context: arguments[0],
                    backend_handle,
                    buffer_address: 0,
                    channel_count,
                    gains: [1.0; 8],
                    parameter,
                },
            );
            state.port_creates = state.port_creates.saturating_add(1);
            let create = state.port_creates;
            write_guest_u64(arguments[2], handle);
            if create <= 8 || create.is_power_of_two() {
                eprintln!(
                    "[REMu][AUDIO2] port create={create} handle={handle} context={} parameter_address={:#x} parameter={parameter:?} channels={channel_count} backend={backend_handle:?} rate={frequency}",
                    arguments[0], arguments[1],
                );
            }
            0
        }
        "IaZXJ9M79uo" => {
            let existed = state.users.remove(&arguments[0]);
            state.user_destroys = state.user_destroys.saturating_add(1);
            let destroy = state.user_destroys;
            if destroy <= 8 || destroy.is_power_of_two() {
                eprintln!(
                    "[REMu][AUDIO2] user destroy={destroy} handle={} existed={existed}",
                    arguments[0]
                );
            }
            0
        }
        "cd+Rtw+D1x8" => {
            let removed = state.ports.remove(&arguments[0]);
            state.port_destroys = state.port_destroys.saturating_add(1);
            let destroy = state.port_destroys;
            if destroy <= 8 || destroy.is_power_of_two() {
                eprintln!(
                    "[REMu][AUDIO2] port destroy={destroy} handle={} existed={} parameter={:?}",
                    arguments[0],
                    removed.is_some(),
                    removed.map(|port| port.parameter),
                );
            }
            if let Some(backend_handle) = removed.and_then(|port| port.backend_handle) {
                let _ = AUDIO_OUT
                    .get_or_init(|| Mutex::new(audio::AudioOutState::default()))
                    .lock()
                    .unwrap_or_else(std::sync::PoisonError::into_inner)
                    .close(backend_handle);
            }
            0
        }
        "on6ZH7Abo10" => {
            state.contexts.remove(&arguments[0]);
            let backend_handles = state
                .ports
                .extract_if(.., |_, port| port.context == arguments[0])
                .filter_map(|(_, port)| port.backend_handle)
                .collect::<Vec<_>>();
            let mut audio = AUDIO_OUT
                .get_or_init(|| Mutex::new(audio::AudioOutState::default()))
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            for handle in backend_handles {
                let _ = audio.close(handle);
            }
            0
        }
        "8XTArSPyWHk" => {
            if !state.ports.contains_key(&arguments[0]) || arguments[1] == 0 {
                return sce_kernel_error(SCE_KERNEL_ERROR_INVALID_ARGUMENT);
            }
            state.attribute_updates = state.attribute_updates.saturating_add(1);
            let count = arguments[2].min(16);
            let Some(attribute_bytes) = usize::try_from(count)
                .ok()
                .and_then(|count| count.checked_mul(std::mem::size_of::<SceAudioOut2Attribute>()))
            else {
                return sce_kernel_error(SCE_KERNEL_ERROR_INVALID_ARGUMENT);
            };
            if !process_guest_range_is_readable(arguments[1], attribute_bytes) {
                return sce_kernel_error(SCE_KERNEL_ERROR_INVALID_ARGUMENT);
            }
            let mut backend_gain_update = None;
            for index in 0..count {
                let base =
                    arguments[1] + index * std::mem::size_of::<SceAudioOut2Attribute>() as u64;
                // SceAudioOut2Attribute begins with a u32 ID and a separate reserved i32. Reading
                // both as one u64 makes a nonzero reserved word part of the ID and prevents PCM
                // attribute zero from attaching its buffer.
                let attribute = decode_audio_out2_attribute(base);
                match attribute.attribute_id {
                    0 => {
                        if attribute.value == 0
                            || attribute.value_size < std::mem::size_of::<SceAudioOut2Pcm>() as u64
                            || !process_guest_range_is_readable(
                                attribute.value,
                                std::mem::size_of::<SceAudioOut2Pcm>(),
                            )
                        {
                            return sce_kernel_error(SCE_KERNEL_ERROR_INVALID_ARGUMENT);
                        }
                        let pcm = SceAudioOut2Pcm {
                            data: read_guest_u64(attribute.value),
                        };
                        if let Some(port) = state.ports.get_mut(&arguments[0]) {
                            port.buffer_address = pcm.data;
                        }
                    }
                    1 => {
                        if attribute.value == 0
                            || attribute.value_size < std::mem::size_of::<f32>() as u64
                            || !attribute
                                .value_size
                                .is_multiple_of(std::mem::size_of::<f32>() as u64)
                        {
                            return sce_kernel_error(SCE_KERNEL_ERROR_INVALID_ARGUMENT);
                        }
                        let gain_bytes = usize::try_from(
                            attribute
                                .value_size
                                .min((8 * std::mem::size_of::<f32>()) as u64),
                        )
                        .expect("at most eight f32 gains fit usize");
                        if !process_guest_range_is_readable(attribute.value, gain_bytes) {
                            return sce_kernel_error(SCE_KERNEL_ERROR_INVALID_ARGUMENT);
                        }
                        let Some(port) = state.ports.get_mut(&arguments[0]) else {
                            return sce_kernel_error(SCE_KERNEL_ERROR_INVALID_ARGUMENT);
                        };
                        let mut gains = port.gains;
                        for (channel, gain) in gains
                            .iter_mut()
                            .enumerate()
                            .take(gain_bytes / std::mem::size_of::<f32>())
                        {
                            let bits = read_guest_u32(
                                attribute.value + (channel * std::mem::size_of::<f32>()) as u64,
                            );
                            let value = f32::from_bits(bits);
                            if !value.is_finite() {
                                return sce_kernel_error(SCE_KERNEL_ERROR_INVALID_ARGUMENT);
                            }
                            *gain = value;
                        }
                        port.gains = gains;
                        if let Some(backend_handle) = port.backend_handle {
                            backend_gain_update = Some((backend_handle, gains));
                        }
                    }
                    _ => {}
                }
            }
            if let Some((backend_handle, gains)) = backend_gain_update {
                let mut audio = AUDIO_OUT
                    .get_or_init(|| Mutex::new(audio::AudioOutState::default()))
                    .lock()
                    .unwrap_or_else(std::sync::PoisonError::into_inner);
                if let Err(error) = audio.set_channel_gains(backend_handle, gains) {
                    return audio_out_error(error);
                }
            }
            if state.attribute_updates <= 8 || state.attribute_updates.is_power_of_two() {
                let attributes = (0..count)
                    .map(|index| {
                        let base = arguments[1]
                            + index * std::mem::size_of::<SceAudioOut2Attribute>() as u64;
                        let attribute = decode_audio_out2_attribute(base);
                        let value = if attribute.value != 0
                            && attribute.value_size >= 8
                            && process_guest_range_is_readable(attribute.value, 8)
                        {
                            read_guest_u64(attribute.value)
                        } else {
                            0
                        };
                        format!(
                            "id={:#x}/reserved={:#x}/data={:#x}/size={:#x}/value={value:#x}",
                            attribute.attribute_id,
                            attribute.reserved,
                            attribute.value,
                            attribute.value_size,
                        )
                    })
                    .collect::<Vec<_>>()
                    .join(",");
                eprintln!(
                    "[REMu][AUDIO2] attributes update={} port={} count={} [{attributes}]",
                    state.attribute_updates, arguments[0], arguments[2]
                );
            }
            0
        }
        "PE2zHMqLSHs" => {
            let Some(context) = state.contexts.get_mut(&arguments[0]) else {
                return sce_kernel_error(SCE_KERNEL_ERROR_INVALID_ARGUMENT);
            };
            audio_out2_update_context_queue(context, Instant::now());
            state.context_advances = state.context_advances.saturating_add(1);
            let advance = state.context_advances;
            if advance <= 8 || advance.is_power_of_two() {
                eprintln!(
                    "[REMu][AUDIO2] context advance={advance} handle={} args={:#x},{:#x},{:#x},{:#x},{:#x}",
                    arguments[0],
                    arguments[1],
                    arguments[2],
                    arguments[3],
                    arguments[4],
                    arguments[5],
                );
            }
            0
        }
        "aII9h5nli9U" => {
            state.context_pushes = state.context_pushes.saturating_add(1);
            let push = state.context_pushes;
            // Reserve one hardware queue slot. Nonblocking pushes report NOT_READY when the queue
            // is full; blocking pushes wait only until the 48 kHz device clock drains a slot.
            loop {
                let now = Instant::now();
                let Some(context) = state.contexts.get_mut(&arguments[0]) else {
                    return sce_kernel_error(SCE_KERNEL_ERROR_INVALID_ARGUMENT);
                };
                audio_out2_update_context_queue(context, now);
                if context.queued < context.queue_depth {
                    context.queued += 1;
                    break;
                }
                if arguments[1] == 0 {
                    return sce_kernel_error(SCE_AUDIO_OUT2_ERROR_NOT_READY);
                }
                let wait_duration = audio_out2_queue_wait_duration(context, now);
                drop(state);
                if !wait_duration.is_zero() {
                    std::thread::sleep(wait_duration);
                } else {
                    std::thread::yield_now();
                }
                state = AUDIO_OUT2
                    .get_or_init(|| Mutex::new(AudioOut2State::default()))
                    .lock()
                    .unwrap_or_else(std::sync::PoisonError::into_inner);
            }
            let grain_frames = state
                .contexts
                .get(&arguments[0])
                .expect("the context was validated while reserving its queue slot")
                .grain_frames;
            // Attribute zero is the only PCM source. Snapshot exactly one configured grain now,
            // while the guest's pointer is attached to this push, so later Wwise pointer updates
            // cannot mutate an already-submitted block. In particular, context/mastering memory
            // is never searched for plausible sample pointers.
            let mut buffers = Vec::new();
            let mut invalid_pcm = None;
            for (port_handle, port) in state.ports.iter().filter(|(_, port)| {
                port.context == arguments[0]
                    && port.buffer_address != 0
                    && port.backend_handle.is_some()
            }) {
                let Some(byte_length) = usize::try_from(grain_frames)
                    .ok()
                    .and_then(|frames| frames.checked_mul(usize::from(port.channel_count)))
                    .and_then(|samples| samples.checked_mul(std::mem::size_of::<f32>()))
                else {
                    invalid_pcm = Some((
                        *port_handle,
                        port.buffer_address,
                        None,
                        "grain byte length overflow",
                    ));
                    break;
                };
                if !process_guest_range_has_read_access(port.buffer_address, byte_length) {
                    invalid_pcm = Some((
                        *port_handle,
                        port.buffer_address,
                        Some(byte_length),
                        "PCM range is not readable",
                    ));
                    break;
                }
                let mut snapshot = vec![0_u8; byte_length];
                read_guest_bytes(port.buffer_address, &mut snapshot);
                buffers.push((
                    *port_handle,
                    port.backend_handle
                        .expect("the buffer filter selects only audible backend ports"),
                    port.buffer_address,
                    port.channel_count,
                    snapshot,
                ));
            }
            if let Some((port_handle, address, byte_length, reason)) = invalid_pcm {
                return reject_audio_out2_reserved_pcm(
                    &mut state,
                    arguments[0],
                    push,
                    port_handle,
                    address,
                    byte_length,
                    reason,
                );
            }
            let audible_push = if buffers.is_empty() {
                0
            } else {
                state.audible_context_pushes = state.audible_context_pushes.saturating_add(1);
                state.audible_context_pushes
            };
            for (port, _, address, channels, snapshot) in &buffers {
                trace_audio_out2_pcm(push, audible_push, *port, *address, *channels, snapshot);
            }
            #[cfg(test)]
            state.submitted_pcm.extend(
                buffers
                    .iter()
                    .map(|(port, _, _, _, snapshot)| (*port, snapshot.clone())),
            );
            let context_memory_trace = (trace_audio_out2_memory()
                && trace_logarithmic_sequence(push, 8))
            .then(|| {
                let context = state.contexts.get(&arguments[0]).copied()?;
                let channels = state
                    .ports
                    .values()
                    .filter(|port| port.context == arguments[0])
                    .map(|port| port.channel_count)
                    .collect::<Vec<_>>();
                (!channels.is_empty()).then_some((context, channels, state.mastering_parameters))
            })
            .flatten();
            if push <= 8 || push.is_power_of_two() {
                eprintln!(
                    "[REMu][AUDIO2] context push={push} handle={} buffers={} args={:#x},{:#x},{:#x},{:#x},{:#x}",
                    arguments[0],
                    buffers.len(),
                    arguments[1],
                    arguments[2],
                    arguments[3],
                    arguments[4],
                    arguments[5],
                );
            }
            drop(state);
            if let Some((context, channels, mastering_parameters)) = context_memory_trace {
                trace_audio_out2_context_memory(context, push, &channels);
                trace_audio_out2_mastering_memory(mastering_parameters, push);
            }

            let mut audio = AUDIO_OUT
                .get_or_init(|| Mutex::new(audio::AudioOutState::default()))
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            let mut backend_error = None;
            for (_, handle, _, _, buffer) in buffers {
                match audio.output_realtime([handle, buffer.as_ptr() as u64, 0, 0, 0, 0]) {
                    Ok(_) => {}
                    Err(error) => {
                        backend_error = Some((handle, error));
                        break;
                    }
                }
            }
            drop(audio);
            if let Some((handle, error)) = backend_error {
                // The console queue reservation and host submission form one operation. A host
                // validation/allocation failure must not leak a queue slot and make later valid
                // pushes report NOT_READY until the virtual device clock happens to drain it.
                let mut state = AUDIO_OUT2
                    .get_or_init(|| Mutex::new(AudioOut2State::default()))
                    .lock()
                    .unwrap_or_else(std::sync::PoisonError::into_inner);
                if let Some(context) = state.contexts.get_mut(&arguments[0]) {
                    context.queued = context.queued.saturating_sub(1);
                }
                eprintln!(
                    "[REMu][AUDIO2] context push={push} backend={handle} failed: {error:?}; released reserved queue slot"
                );
                return audio_out_error(error);
            }
            0
        }
        _ => unreachable!("caller filters AudioOut2 imports"),
    }
}
