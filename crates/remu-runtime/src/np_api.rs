use crate::{
    ProbeDispatchState, guest_memset, low_i32, low_u32, np, read_guest_cstring, read_guest_u64,
    sce_kernel_error, write_guest_u8, write_guest_u32, write_guest_u64, write_output_i32,
};

fn np_error(result: Result<(), u32>) -> u64 {
    result.map_or_else(sce_kernel_error, |()| 0)
}

fn read_sized_guest_parameters(
    address: u64,
    minimum_size: usize,
    maximum_size: usize,
) -> Option<Vec<u8>> {
    if address == 0 {
        return None;
    }
    let size = usize::try_from(read_guest_u64(address)).ok()?;
    if !(minimum_size..=maximum_size).contains(&size) {
        return None;
    }
    // SAFETY: the ABI's leading size field describes a readable initialization structure owned by
    // the caller. The bounded copy means NP state never retains a guest pointer past this call.
    Some(unsafe { std::slice::from_raw_parts(address as *const u8, size) }.to_vec())
}

pub(crate) fn np_universal_data_initialize(parameter: u64, state: &mut ProbeDispatchState) -> u64 {
    let Some(parameters) = read_sized_guest_parameters(parameter, 16, 0x1000) else {
        return sce_kernel_error(np::UDS_ERROR_INVALID_ARGUMENT);
    };
    np_error(state.np_universal_data.initialize(parameters))
}

pub(crate) fn np_universal_data_terminate(state: &mut ProbeDispatchState) -> u64 {
    np_error(state.np_universal_data.terminate())
}

pub(crate) fn np_universal_data_create_context(
    arguments: [u64; 6],
    state: &mut ProbeDispatchState,
) -> u64 {
    if arguments[0] == 0 {
        return sce_kernel_error(np::UDS_ERROR_INVALID_ARGUMENT);
    }
    match state
        .np_universal_data
        .create_context(low_i32(arguments[1]), arguments[2])
    {
        Ok(context) => write_output_i32(arguments[0], context),
        Err(error) => sce_kernel_error(error),
    }
}

pub(crate) fn np_universal_data_destroy_context(
    context: u64,
    state: &mut ProbeDispatchState,
) -> u64 {
    np_error(state.np_universal_data.destroy_context(low_i32(context)))
}

pub(crate) fn np_universal_data_create_handle(
    arguments: [u64; 6],
    state: &mut ProbeDispatchState,
) -> u64 {
    if arguments[0] == 0 {
        return sce_kernel_error(np::UDS_ERROR_INVALID_ARGUMENT);
    }
    match state.np_universal_data.create_handle() {
        Ok(handle) => write_output_i32(arguments[0], handle),
        Err(error) => sce_kernel_error(error),
    }
}

pub(crate) fn np_universal_data_abort_handle(handle: u64, state: &mut ProbeDispatchState) -> u64 {
    np_error(state.np_universal_data.abort_handle(low_i32(handle)))
}

pub(crate) fn np_universal_data_destroy_handle(handle: u64, state: &mut ProbeDispatchState) -> u64 {
    np_error(state.np_universal_data.destroy_handle(low_i32(handle)))
}

pub(crate) fn np_universal_data_register_context(
    arguments: [u64; 6],
    state: &mut ProbeDispatchState,
) -> u64 {
    np_error(
        state
            .np_universal_data
            .register_context(low_i32(arguments[0]), low_i32(arguments[1])),
    )
}

pub(crate) fn np_universal_data_create_event(
    arguments: [u64; 6],
    state: &mut ProbeDispatchState,
) -> u64 {
    if arguments[2] == 0 || arguments[3] == 0 {
        return sce_kernel_error(np::UDS_ERROR_INVALID_ARGUMENT);
    }
    let Some(name) = read_guest_cstring(arguments[0]) else {
        return sce_kernel_error(np::UDS_ERROR_INVALID_EVENT_NAME);
    };
    match state.np_universal_data.create_event(name, arguments[1]) {
        Ok((event, root_object)) => {
            write_guest_u64(arguments[2], event);
            write_guest_u64(arguments[3], root_object);
            0
        }
        Err(error) => sce_kernel_error(error),
    }
}

pub(crate) fn np_universal_data_destroy_event(event: u64, state: &mut ProbeDispatchState) -> u64 {
    np_error(state.np_universal_data.destroy_event(event))
}

pub(crate) fn np_universal_data_create_property_object(
    output: u64,
    state: &mut ProbeDispatchState,
) -> u64 {
    if output == 0 {
        return sce_kernel_error(np::UDS_ERROR_INVALID_ARGUMENT);
    }
    match state.np_universal_data.create_property_object() {
        Ok(object) => {
            write_guest_u64(output, object);
            0
        }
        Err(error) => sce_kernel_error(error),
    }
}

/// `sceNpUniversalDataSystemCreateEventPropertyArray(arrayOut)`.
///
/// Observed call sites pass extra unused register values after the out-pointer (same pattern as
/// Initialize); only the first argument is the array handle slot.
pub(crate) fn np_universal_data_create_property_array(
    output: u64,
    state: &mut ProbeDispatchState,
) -> u64 {
    if output == 0 {
        return sce_kernel_error(np::UDS_ERROR_INVALID_ARGUMENT);
    }
    match state.np_universal_data.create_property_array(0) {
        Ok(array) => {
            write_guest_u64(output, array);
            0
        }
        Err(error) => sce_kernel_error(error),
    }
}

pub(crate) fn np_universal_data_destroy_property_array(
    array: u64,
    state: &mut ProbeDispatchState,
) -> u64 {
    np_error(state.np_universal_data.destroy_property_array(array))
}

pub(crate) fn np_universal_data_object_set_i32(
    arguments: [u64; 6],
    state: &mut ProbeDispatchState,
) -> u64 {
    let Some(key) = read_guest_cstring(arguments[1]) else {
        return sce_kernel_error(np::UDS_ERROR_INVALID_EVENT_PROPERTY_KEY);
    };
    np_error(
        state
            .np_universal_data
            .object_set_i32(arguments[0], key, low_i32(arguments[2])),
    )
}

pub(crate) fn np_universal_data_object_set_string(
    arguments: [u64; 6],
    state: &mut ProbeDispatchState,
) -> u64 {
    let Some(key) = read_guest_cstring(arguments[1]) else {
        return sce_kernel_error(np::UDS_ERROR_INVALID_EVENT_PROPERTY_KEY);
    };
    let Some(value) = read_guest_cstring(arguments[2]) else {
        return sce_kernel_error(np::UDS_ERROR_INVALID_EVENT_PROPERTY_VALUE);
    };
    np_error(
        state
            .np_universal_data
            .object_set_string(arguments[0], key, value),
    )
}

/// `sceNpUniversalDataSystemEventPropertyObjectSetArray(object, key, value, valueOut)`.
///
/// `value` may already be a live array from `CreateEventPropertyArray`. When both `value` and
/// `valueOut` are null the call is a no-op success, matching the Gen5 cleanup path after a
/// failed create.
pub(crate) fn np_universal_data_object_set_array(
    arguments: [u64; 6],
    state: &mut ProbeDispatchState,
) -> u64 {
    let (object, existing, output) = (arguments[0], arguments[2], arguments[3]);
    if object == 0 {
        return sce_kernel_error(np::UDS_ERROR_INVALID_ARGUMENT);
    }
    if existing == 0 && output == 0 {
        return 0;
    }
    let Some(key) = read_guest_cstring(arguments[1]) else {
        return sce_kernel_error(np::UDS_ERROR_INVALID_EVENT_PROPERTY_KEY);
    };
    let existing = (existing != 0).then_some(existing);
    match state
        .np_universal_data
        .object_set_array(object, key, existing)
    {
        Ok(array) => {
            if output != 0 {
                write_guest_u64(output, array);
            }
            0
        }
        Err(error) => sce_kernel_error(error),
    }
}

pub(crate) fn np_universal_data_array_set_string(
    arguments: [u64; 6],
    state: &mut ProbeDispatchState,
) -> u64 {
    let Some(value) = read_guest_cstring(arguments[1]) else {
        return sce_kernel_error(np::UDS_ERROR_INVALID_EVENT_PROPERTY_VALUE);
    };
    np_error(
        state
            .np_universal_data
            .array_append_string(arguments[0], value),
    )
}

pub(crate) fn np_universal_data_post_event(
    arguments: [u64; 6],
    state: &mut ProbeDispatchState,
) -> u64 {
    match state.np_universal_data.post_event(
        low_i32(arguments[0]),
        low_i32(arguments[1]),
        arguments[2],
        arguments[3],
    ) {
        Ok(posted) => {
            eprintln!(
                "[REMu][NP] committed universal-data event sequence={} context={} user={:#x} name={:?} bytes={} options={:#x}/{:#x}/{:#x}",
                posted.sequence,
                posted.context,
                posted.user_id,
                posted.event_name,
                posted.payload_size,
                posted.context_option,
                posted.event_option,
                posted.post_option,
            );
            0
        }
        Err(error) => sce_kernel_error(error),
    }
}

pub(crate) fn np_game_intent_initialize(parameter: u64, state: &mut ProbeDispatchState) -> u64 {
    let Some(parameters) = read_sized_guest_parameters(parameter, 0x28, 0x1000) else {
        return sce_kernel_error(np::GAME_INTENT_ERROR_INVALID_ARGUMENT);
    };
    np_error(state.np_game_intent.initialize(parameters))
}

pub(crate) fn np_game_intent_receive(output: u64, state: &mut ProbeDispatchState) -> u64 {
    const GAME_INTENT_OUTPUT_SIZE: usize = 0x4140;
    const INTENT_TYPE_OFFSET: u64 = 0x0c;
    const INTENT_TYPE_CAPACITY: usize = 0x100;
    const PROPERTY_OBJECT_OFFSET: u64 = 0x134;
    if output == 0 {
        return sce_kernel_error(np::GAME_INTENT_ERROR_INVALID_ARGUMENT);
    }
    let Ok(output_size) = usize::try_from(read_guest_u64(output)) else {
        return sce_kernel_error(np::GAME_INTENT_ERROR_INVALID_ARGUMENT);
    };
    if output_size < GAME_INTENT_OUTPUT_SIZE {
        return sce_kernel_error(np::GAME_INTENT_ERROR_INSUFFICIENT_BUFFER);
    }
    let intent = match state.np_game_intent.receive() {
        Ok(intent) => intent,
        Err(error) => return sce_kernel_error(error),
    };
    if intent.intent_type.is_empty()
        || intent.intent_type.len() >= INTENT_TYPE_CAPACITY
        || intent.intent_type.as_bytes().contains(&0)
    {
        return sce_kernel_error(np::GAME_INTENT_ERROR_INVALID_ARGUMENT);
    }
    guest_memset(
        output + 8,
        0,
        u64::try_from(GAME_INTENT_OUTPUT_SIZE - 8).expect("fixed intent size fits u64"),
    );
    write_guest_u32(output + 8, 0);
    for (index, byte) in intent.intent_type.bytes().enumerate() {
        write_guest_u8(
            output + INTENT_TYPE_OFFSET + u64::try_from(index).expect("intent type index fits u64"),
            byte,
        );
    }
    let property_object = output + PROPERTY_OBJECT_OFFSET;
    state
        .np_game_intent
        .retain_received_properties(property_object, intent.properties);
    0
}

pub(crate) fn np_game_intent_get_property_string(
    arguments: [u64; 6],
    state: &mut ProbeDispatchState,
) -> u64 {
    if arguments[2] == 0 || arguments[3] == 0 {
        return sce_kernel_error(np::GAME_INTENT_ERROR_INVALID_ARGUMENT);
    }
    let Some(key) = read_guest_cstring(arguments[1]) else {
        return sce_kernel_error(np::GAME_INTENT_ERROR_INVALID_ARGUMENT);
    };
    let value = match state.np_game_intent.property(arguments[0], &key) {
        Ok(value) => value.to_owned(),
        Err(error) => return sce_kernel_error(error),
    };
    let Some(required) = value.len().checked_add(1) else {
        return sce_kernel_error(np::GAME_INTENT_ERROR_INSUFFICIENT_BUFFER);
    };
    if u64::try_from(required).unwrap_or(u64::MAX) > arguments[3] {
        return sce_kernel_error(np::GAME_INTENT_ERROR_INSUFFICIENT_BUFFER);
    }
    for (index, byte) in value.bytes().chain(std::iter::once(0)).enumerate() {
        write_guest_u8(
            arguments[2] + u64::try_from(index).expect("property string index fits u64"),
            byte,
        );
    }
    0
}

pub(crate) fn np_trophy_create_context(arguments: [u64; 6], state: &mut ProbeDispatchState) -> u64 {
    if arguments[0] == 0 {
        return sce_kernel_error(np::TROPHY2_ERROR_INVALID_ARGUMENT);
    }
    let context = state
        .np_trophy
        .create_context(low_i32(arguments[1]), arguments[2]);
    write_output_i32(arguments[0], context)
}

pub(crate) fn np_trophy_destroy_context(context: u64, state: &mut ProbeDispatchState) -> u64 {
    np_error(state.np_trophy.destroy_context(low_i32(context)))
}

pub(crate) fn np_trophy_create_handle(arguments: [u64; 6], state: &mut ProbeDispatchState) -> u64 {
    if arguments[0] == 0 {
        return sce_kernel_error(np::TROPHY2_ERROR_INVALID_ARGUMENT);
    }
    let handle = state.np_trophy.create_handle();
    write_output_i32(arguments[0], handle)
}

pub(crate) fn np_trophy_destroy_handle(handle: u64, state: &mut ProbeDispatchState) -> u64 {
    np_error(state.np_trophy.destroy_handle(low_i32(handle)))
}

pub(crate) fn np_trophy_abort_handle(handle: u64, state: &mut ProbeDispatchState) -> u64 {
    np_error(state.np_trophy.abort_handle(low_i32(handle)))
}

pub(crate) fn np_trophy_register_context(
    arguments: [u64; 6],
    state: &mut ProbeDispatchState,
) -> u64 {
    np_error(
        state
            .np_trophy
            .register_context(low_i32(arguments[0]), low_i32(arguments[1])),
    )
}

/// `sceNpTrophy2GetGameInfo(context, handle, gameInfo)`.
///
/// This host has no PSN trophy metadata. The call validates the registered context and handle,
/// zeroes the game-info record so the guest reads empty titles and zero trophy counts, and shows a
/// one-shot host toast so the offline nature is visible rather than silent.
pub(crate) fn np_trophy_get_game_info(arguments: [u64; 6], state: &mut ProbeDispatchState) -> u64 {
    // Observed layout leaves ~0x98 bytes before the next stack slot the title keeps live.
    const GAME_INFO_SIZE: u64 = 0x90;

    let (context, handle, game_info) = (low_i32(arguments[0]), low_i32(arguments[1]), arguments[2]);
    if game_info == 0 {
        return sce_kernel_error(np::TROPHY2_ERROR_INVALID_ARGUMENT);
    }
    if let Err(error) = state.np_trophy.require_registered(context, handle) {
        return np_error(Err(error));
    }
    remu_gpu::note_inert("np:Trophy2GetGameInfo");
    guest_memset(game_info, 0, GAME_INFO_SIZE);
    crate::publish_host_toast(
        remu_gpu::HostDialogKind::Trophy,
        "Trophies",
        "Trophy metadata is unavailable offline.",
    );
    0
}

/// `sceNpTrophy2GetTrophyInfoArray(context, handle, offset, limit, details, data, count)`.
///
/// Offline hosts have no trophy catalogue. A zero `limit` (or missing output buffers) reports
/// `count=0` after validating the registered context/handle, which is what Octopath asks for
/// during boot. Inventing placeholder trophies would misrepresent unlock state.
pub(crate) fn np_trophy_get_trophy_info_array(
    arguments: [u64; 6],
    stack_arguments: [u64; 6],
    state: &mut ProbeDispatchState,
) -> u64 {
    let (context, handle, _offset, limit, details, data) = (
        low_i32(arguments[0]),
        low_i32(arguments[1]),
        low_u32(arguments[2]),
        low_u32(arguments[3]),
        arguments[4],
        arguments[5],
    );
    let count = stack_arguments[0];
    if let Err(error) = state.np_trophy.require_registered(context, handle) {
        return np_error(Err(error));
    }
    remu_gpu::note_inert("np:Trophy2GetTrophyInfoArray");
    let _ = (details, data, limit);
    if count != 0 {
        write_guest_u32(count, 0);
    }
    0
}

pub(crate) fn np_trophy_register_unlock_callback(
    arguments: [u64; 6],
    state: &mut ProbeDispatchState,
) -> u64 {
    np_error(
        state
            .np_trophy
            .register_unlock_callback(arguments[0], arguments[1]),
    )
}

pub(crate) fn np_trophy_unregister_unlock_callback(state: &mut ProbeDispatchState) -> u64 {
    np_error(state.np_trophy.unregister_unlock_callback())
}
