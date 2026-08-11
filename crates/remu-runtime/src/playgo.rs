use remu_memory::GUEST_PAGE_SIZE;

use crate::{
    ProbeDispatchState, low_u32, process_guest_range_has_read_access, read_guest_u16,
    sce_kernel_error, write_guest_u8, write_guest_u16, write_guest_u32, write_guest_u64,
};

const PLAYGO_ERROR_NOT_INITIALIZED: u32 = 0x80b2_0005;
const PLAYGO_ERROR_ALREADY_INITIALIZED: u32 = 0x80b2_0006;
const PLAYGO_ERROR_ALREADY_STARTED: u32 = 0x80b2_0007;
const PLAYGO_ERROR_NOT_STARTED: u32 = 0x80b2_0008;
const PLAYGO_ERROR_BAD_HANDLE: u32 = 0x80b2_0009;
const PLAYGO_ERROR_BAD_POINTER: u32 = 0x80b2_000a;
const PLAYGO_ERROR_BAD_SIZE: u32 = 0x80b2_000b;
const PLAYGO_ERROR_BAD_CHUNK_ID: u32 = 0x80b2_000c;
const PLAYGO_CHUNK_INDEX_MAX: u32 = 1_000;
/// The only handle the console hands out for the running application's PlayGo package.
const PLAYGO_HANDLE: u32 = 1;
/// Chunk is resident on local storage with fast access. REmu runs from a fully installed local
/// image, so every chunk the title asks about is already present.
const PLAYGO_LOCUS_LOCAL_FAST: u8 = 3;
const PLAYGO_INSTALL_SPEED_FULL: u32 = 2;

/// Reads the plain-text package description shipped by unpacked title dumps.
///
/// Retail systems obtain this list from package metadata. REmu's extracted app directories do not
/// consistently retain the binary package table, but development and preservation dumps commonly
/// include `playgo-chunkdefs.xml`. Every declared `<chunk>` is an ABI-visible identifier, including
/// empty placeholders marked `enabled="false"`; `<chunkset>` elements are grouping metadata rather
/// than chunks.
fn discover_play_go_chunk_ids(state: &ProbeDispatchState) -> Vec<u16> {
    let Ok(xml) = std::fs::read_to_string(state.game_root.join("playgo-chunkdefs.xml")) else {
        return Vec::new();
    };
    let mut ids = Vec::new();
    let mut cursor = xml.as_str();
    while let Some(offset) = cursor.find("<chunk") {
        cursor = &cursor[offset + "<chunk".len()..];
        let Some(first) = cursor.as_bytes().first() else {
            break;
        };
        if !first.is_ascii_whitespace() {
            continue;
        }
        let Some(end) = cursor.find('>') else {
            break;
        };
        let tag = &cursor[..end];
        cursor = &cursor[end + 1..];
        let id = xml_attribute(tag, "id").and_then(|value| value.parse::<u16>().ok());
        if let Some(id) = id {
            ids.push(id);
        }
    }
    ids.sort_unstable();
    ids.dedup();
    ids
}

fn xml_attribute<'a>(tag: &'a str, name: &str) -> Option<&'a str> {
    tag.split_ascii_whitespace().find_map(|field| {
        let value = field.strip_prefix(name)?.strip_prefix('=')?;
        value
            .strip_prefix('"')
            .and_then(|value| value.strip_suffix('"'))
            .or_else(|| {
                value
                    .strip_prefix('\'')
                    .and_then(|value| value.strip_suffix('\''))
            })
    })
}

fn package_chunk_ids(state: &ProbeDispatchState) -> &[u16] {
    if state.services.play_go_chunk_ids.is_empty() {
        &[0]
    } else {
        &state.services.play_go_chunk_ids
    }
}

/// Reports whether PlayGo is initialized, rejecting the double-initialize the ABI forbids.
pub(crate) fn play_go_initialize(state: &mut ProbeDispatchState) -> u64 {
    if state.services.play_go_initialized {
        return sce_kernel_error(PLAYGO_ERROR_ALREADY_INITIALIZED);
    }
    state.services.play_go_chunk_ids = discover_play_go_chunk_ids(state);
    state.services.play_go_initialized = true;
    0
}

pub(crate) fn play_go_terminate(state: &mut ProbeDispatchState) -> u64 {
    if !state.services.play_go_initialized {
        return sce_kernel_error(PLAYGO_ERROR_NOT_INITIALIZED);
    }
    state.services.play_go_initialized = false;
    state.services.play_go_open = false;
    state.services.play_go_chunk_ids.clear();
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
    if !process_guest_range_has_read_access(output, 4) {
        return sce_kernel_error(PLAYGO_ERROR_BAD_POINTER);
    }
    if state.services.play_go_open {
        return sce_kernel_error(PLAYGO_ERROR_ALREADY_STARTED);
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
    output_size: usize,
    state: &ProbeDispatchState,
) -> Result<u32, u64> {
    if !state.services.play_go_initialized {
        return Err(sce_kernel_error(PLAYGO_ERROR_NOT_INITIALIZED));
    }
    if low_u32(handle) != PLAYGO_HANDLE {
        return Err(sce_kernel_error(PLAYGO_ERROR_BAD_HANDLE));
    }
    if !state.services.play_go_open {
        return Err(sce_kernel_error(PLAYGO_ERROR_NOT_STARTED));
    }
    if chunk_ids == 0 || output == 0 {
        return Err(sce_kernel_error(PLAYGO_ERROR_BAD_POINTER));
    }
    let entries = low_u32(entries);
    if entries == 0 || entries > PLAYGO_CHUNK_INDEX_MAX {
        return Err(sce_kernel_error(PLAYGO_ERROR_BAD_SIZE));
    }
    let chunk_id_bytes = usize::try_from(entries)
        .ok()
        .and_then(|entries| entries.checked_mul(std::mem::size_of::<u16>()))
        .ok_or_else(|| sce_kernel_error(PLAYGO_ERROR_BAD_SIZE))?;
    if !process_guest_range_has_read_access(chunk_ids, chunk_id_bytes)
        || !process_guest_range_has_read_access(output, output_size)
    {
        return Err(sce_kernel_error(PLAYGO_ERROR_BAD_POINTER));
    }
    for entry in 0..u64::from(entries) {
        let chunk_id = read_guest_u16(chunk_ids + entry * 2);
        if !package_chunk_ids(state).contains(&chunk_id) {
            return Err(sce_kernel_error(PLAYGO_ERROR_BAD_CHUNK_ID));
        }
    }
    Ok(entries)
}

pub(crate) fn play_go_get_locus(arguments: [u64; 6], state: &ProbeDispatchState) -> u64 {
    let entries = match play_go_entry_query(
        arguments[0],
        arguments[1],
        arguments[2],
        arguments[3],
        usize::try_from(low_u32(arguments[2])).unwrap_or(usize::MAX),
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
        16,
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
        8,
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
    if !state.services.play_go_open {
        return sce_kernel_error(PLAYGO_ERROR_NOT_STARTED);
    }
    if arguments[1] == 0 || !process_guest_range_has_read_access(arguments[1], 4) {
        return sce_kernel_error(PLAYGO_ERROR_BAD_POINTER);
    }
    write_guest_u32(arguments[1], PLAYGO_INSTALL_SPEED_FULL);
    0
}

/// Reports the chunk identifiers belonging to the installed image.
///
/// Installed titles expose the enabled identifiers from their package metadata. If an extracted
/// dump has lost that metadata, the mandatory launch chunk (zero) remains available.
pub(crate) fn play_go_get_chunk_id(arguments: [u64; 6], state: &ProbeDispatchState) -> u64 {
    let (output, entries, count_output) = (arguments[1], low_u32(arguments[2]), arguments[3]);
    if !state.services.play_go_initialized {
        return sce_kernel_error(PLAYGO_ERROR_NOT_INITIALIZED);
    }
    if low_u32(arguments[0]) != PLAYGO_HANDLE {
        return sce_kernel_error(PLAYGO_ERROR_BAD_HANDLE);
    }
    if !state.services.play_go_open {
        return sce_kernel_error(PLAYGO_ERROR_NOT_STARTED);
    }
    if output == 0 || count_output == 0 || !process_guest_range_has_read_access(count_output, 4) {
        return sce_kernel_error(PLAYGO_ERROR_BAD_POINTER);
    }
    if entries == 0 || entries > PLAYGO_CHUNK_INDEX_MAX {
        return sce_kernel_error(PLAYGO_ERROR_BAD_SIZE);
    }
    let chunks = package_chunk_ids(state);
    let written = usize::try_from(entries)
        .unwrap_or(usize::MAX)
        .min(chunks.len());
    let Some(output_size) = written.checked_mul(std::mem::size_of::<u16>()) else {
        return sce_kernel_error(PLAYGO_ERROR_BAD_SIZE);
    };
    if !process_guest_range_has_read_access(output, output_size) {
        return sce_kernel_error(PLAYGO_ERROR_BAD_POINTER);
    }
    for (entry, chunk_id) in chunks.iter().copied().take(written).enumerate() {
        write_guest_u16(output + (entry as u64) * 2, chunk_id);
    }
    write_guest_u32(count_output, u32::try_from(written).unwrap_or(u32::MAX));
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
    if !state.services.play_go_open {
        return sce_kernel_error(PLAYGO_ERROR_NOT_STARTED);
    }
    if arguments[1] == 0 || !process_guest_range_has_read_access(arguments[1], 8) {
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
    if !state.services.play_go_open {
        return sce_kernel_error(PLAYGO_ERROR_NOT_STARTED);
    }
    0
}
