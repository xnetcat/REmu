use remu_memory::GUEST_PAGE_SIZE;

use crate::{
    ProbeDispatchState, low_u32, sce_kernel_error, write_guest_u8, write_guest_u32, write_guest_u64,
};

const PLAYGO_ERROR_NOT_INITIALIZED: u32 = 0x80b2_0002;
const PLAYGO_ERROR_ALREADY_INITIALIZED: u32 = 0x80b2_0004;
const PLAYGO_ERROR_BAD_HANDLE: u32 = 0x80b2_0006;
const PLAYGO_ERROR_BAD_POINTER: u32 = 0x80b2_0008;
const PLAYGO_ERROR_BAD_SIZE: u32 = 0x80b2_0009;
/// The only handle the console hands out for the running application's PlayGo package.
const PLAYGO_HANDLE: u32 = 1;
/// Chunk is resident on local storage with fast access. REmu runs from a fully installed local
/// image, so every chunk the title asks about is already present.
const PLAYGO_LOCUS_LOCAL_FAST: u8 = 3;
const PLAYGO_INSTALL_SPEED_FULL: u32 = 1;

/// Reports whether PlayGo is initialized, rejecting the double-initialize the ABI forbids.
pub(crate) fn play_go_initialize(state: &mut ProbeDispatchState) -> u64 {
    if state.services.play_go_initialized {
        return sce_kernel_error(PLAYGO_ERROR_ALREADY_INITIALIZED);
    }
    state.services.play_go_initialized = true;
    0
}

pub(crate) fn play_go_terminate(state: &mut ProbeDispatchState) -> u64 {
    if !state.services.play_go_initialized {
        return sce_kernel_error(PLAYGO_ERROR_NOT_INITIALIZED);
    }
    state.services.play_go_initialized = false;
    state.services.play_go_open = false;
    0
}

pub(crate) fn play_go_open(arguments: [u64; 6], state: &mut ProbeDispatchState) -> u64 {
    let output = arguments[0];
    if !state.services.play_go_initialized {
        return sce_kernel_error(PLAYGO_ERROR_NOT_INITIALIZED);
    }
    if output == 0 {
        return sce_kernel_error(PLAYGO_ERROR_BAD_POINTER);
    }
    state.services.play_go_open = true;
    write_guest_u32(output, PLAYGO_HANDLE);
    0
}

pub(crate) fn play_go_close(handle: u64, state: &mut ProbeDispatchState) -> u64 {
    if low_u32(handle) != PLAYGO_HANDLE || !state.services.play_go_open {
        return sce_kernel_error(PLAYGO_ERROR_BAD_HANDLE);
    }
    state.services.play_go_open = false;
    0
}

/// Validates the handle and entry count shared by every per-chunk PlayGo query.
fn play_go_entry_query(
    handle: u64,
    chunk_ids: u64,
    entries: u64,
    output: u64,
    state: &ProbeDispatchState,
) -> Result<u32, u64> {
    if !state.services.play_go_initialized {
        return Err(sce_kernel_error(PLAYGO_ERROR_NOT_INITIALIZED));
    }
    if low_u32(handle) != PLAYGO_HANDLE {
        return Err(sce_kernel_error(PLAYGO_ERROR_BAD_HANDLE));
    }
    if chunk_ids == 0 || output == 0 {
        return Err(sce_kernel_error(PLAYGO_ERROR_BAD_POINTER));
    }
    let entries = low_u32(entries);
    if entries == 0 {
        return Err(sce_kernel_error(PLAYGO_ERROR_BAD_SIZE));
    }
    Ok(entries)
}

pub(crate) fn play_go_get_locus(arguments: [u64; 6], state: &ProbeDispatchState) -> u64 {
    let entries = match play_go_entry_query(
        arguments[0],
        arguments[1],
        arguments[2],
        arguments[3],
        state,
    ) {
        Ok(entries) => entries,
        Err(error) => return error,
    };
    for entry in 0..u64::from(entries) {
        // ScePlayGoLocus is an int8_t enum and `out_loci` is a tightly packed byte array.
        // Writing dwords here corrupts the caller's adjacent stack fields; Silent Hill keeps
        // this single-byte result immediately beside other PlayGo temporaries.
        write_guest_u8(arguments[3] + entry, PLAYGO_LOCUS_LOCAL_FAST);
    }
    0
}

/// Reports installation progress. A locally installed image is always fully downloaded, so the
/// progressed and total sizes are equal and non-zero; a zero total reads as "nothing installed".
pub(crate) fn play_go_get_progress(arguments: [u64; 6], state: &ProbeDispatchState) -> u64 {
    let entries = match play_go_entry_query(
        arguments[0],
        arguments[1],
        arguments[2],
        arguments[3],
        state,
    ) {
        Ok(entries) => entries,
        Err(error) => return error,
    };
    // The console reports byte counts per requested chunk; the exact size is unobservable to the
    // title beyond progressed == total, which is what "installed" means.
    let total = u64::from(entries) * GUEST_PAGE_SIZE;
    write_guest_u64(arguments[3], total);
    write_guest_u64(arguments[3] + 8, total);
    0
}

pub(crate) fn play_go_get_eta(arguments: [u64; 6], state: &ProbeDispatchState) -> u64 {
    let _ = match play_go_entry_query(
        arguments[0],
        arguments[1],
        arguments[2],
        arguments[3],
        state,
    ) {
        Ok(entries) => entries,
        Err(error) => return error,
    };
    // Nothing remains to be downloaded.
    write_guest_u64(arguments[3], 0);
    0
}

pub(crate) fn play_go_get_install_speed(arguments: [u64; 6], state: &ProbeDispatchState) -> u64 {
    if !state.services.play_go_initialized {
        return sce_kernel_error(PLAYGO_ERROR_NOT_INITIALIZED);
    }
    if low_u32(arguments[0]) != PLAYGO_HANDLE {
        return sce_kernel_error(PLAYGO_ERROR_BAD_HANDLE);
    }
    if arguments[1] == 0 {
        return sce_kernel_error(PLAYGO_ERROR_BAD_POINTER);
    }
    write_guest_u32(arguments[1], PLAYGO_INSTALL_SPEED_FULL);
    0
}

/// Reports the chunk identifiers belonging to the installed image.
///
/// A locally installed title is delivered as one chunk, so the list contains only chunk zero.
pub(crate) fn play_go_get_chunk_id(arguments: [u64; 6], state: &ProbeDispatchState) -> u64 {
    let (output, entries, count_output) = (arguments[1], low_u32(arguments[2]), arguments[3]);
    if !state.services.play_go_initialized {
        return sce_kernel_error(PLAYGO_ERROR_NOT_INITIALIZED);
    }
    if low_u32(arguments[0]) != PLAYGO_HANDLE {
        return sce_kernel_error(PLAYGO_ERROR_BAD_HANDLE);
    }
    if output == 0 || count_output == 0 {
        return sce_kernel_error(PLAYGO_ERROR_BAD_POINTER);
    }
    if entries == 0 {
        return sce_kernel_error(PLAYGO_ERROR_BAD_SIZE);
    }
    write_guest_u32(output, 0);
    write_guest_u32(count_output, 1);
    0
}

/// Reports the language mask selected for the package. Zero means no language filtering.
pub(crate) fn play_go_get_language_mask(arguments: [u64; 6], state: &ProbeDispatchState) -> u64 {
    if !state.services.play_go_initialized {
        return sce_kernel_error(PLAYGO_ERROR_NOT_INITIALIZED);
    }
    if low_u32(arguments[0]) != PLAYGO_HANDLE {
        return sce_kernel_error(PLAYGO_ERROR_BAD_HANDLE);
    }
    if arguments[1] == 0 {
        return sce_kernel_error(PLAYGO_ERROR_BAD_POINTER);
    }
    write_guest_u64(arguments[1], 0);
    0
}

/// Accepts a download-ordering or speed request against an already complete installation.
pub(crate) fn play_go_accept_request(handle: u64, state: &ProbeDispatchState) -> u64 {
    if !state.services.play_go_initialized {
        return sce_kernel_error(PLAYGO_ERROR_NOT_INITIALIZED);
    }
    if low_u32(handle) != PLAYGO_HANDLE {
        return sce_kernel_error(PLAYGO_ERROR_BAD_HANDLE);
    }
    0
}
