use std::collections::BTreeMap;
#[cfg(unix)]
use std::os::unix::fs::FileExt;
#[cfg(windows)]
use std::os::windows::fs::FileExt;
use std::sync::{Arc, Mutex, RwLock};

use crate::{
    AMPR_COMMAND_ORDER, AMPR_READ_STAGING_POOL, AmprAddressWrite, AmprCommandBufferState,
    AmprCompletion, AmprReadCommand, KERNEL_EVENT_FILTER_AMPR, KernelQueuedEvent, PROBE_DISPATCH,
    ProbeDispatchState, SCE_KERNEL_ERROR_INVALID_ARGUMENT, SCE_KERNEL_ERROR_MEMORY_FAULT,
    SCE_KERNEL_ERROR_NO_SPACE, SCE_KERNEL_ERROR_NOT_FOUND, SCE_KERNEL_ERROR_OUT_OF_MEMORY,
    current_guest_thread_name, guest_memset, kernel_stat_payload_from_metadata, low_u32,
    notify_kernel_equeue_waiters, process_guest_range_has_read_access, queue_kernel_event,
    read_guest_cstring, read_guest_u64, resolve_probe_guest_path, sce_kernel_error,
    trace_ampr_reads, write_guest_u32, write_guest_u64,
};

const AMPR_ORDER_LOCK_COUNT: usize = 64;
const AMPR_STAGING_POOL_LIMIT: usize = 4;
const AMPR_MAX_RETAINED_STAGING_BYTES: usize = 8 * 1024 * 1024;

fn positional_file_read(
    file: &std::fs::File,
    destination: &mut [u8],
    offset: u64,
) -> std::io::Result<usize> {
    #[cfg(unix)]
    {
        file.read_at(destination, offset)
    }
    #[cfg(windows)]
    {
        file.seek_read(destination, offset)
    }
    #[cfg(not(any(unix, windows)))]
    {
        let _ = (file, destination, offset);
        Err(std::io::Error::from(std::io::ErrorKind::Unsupported))
    }
}

pub(crate) fn ampr_command_order(command_buffer: u64) -> &'static Mutex<()> {
    let locks = AMPR_COMMAND_ORDER
        .get_or_init(|| (0..AMPR_ORDER_LOCK_COUNT).map(|_| Mutex::new(())).collect());
    let index = usize::try_from(command_buffer >> 4).unwrap_or(0) % locks.len();
    &locks[index]
}

fn ampr_destination_has_write_access(
    state: &ProbeDispatchState,
    address: u64,
    length: usize,
) -> bool {
    if length == 0 {
        return true;
    }
    let Ok(length_u64) = u64::try_from(length) else {
        return false;
    };
    let Some(end) = address.checked_add(length_u64) else {
        return false;
    };
    #[cfg(target_os = "macos")]
    if let Some(memory) = state.kernel_memory.as_ref() {
        let start = remu_core::GuestAddress::new(address);
        if memory.contains_virtual_address(start) {
            let mut cursor = address;
            while cursor < end {
                let Some(mapping) = memory.query(remu_core::GuestAddress::new(cursor), false)
                else {
                    return false;
                };
                let mapping_start = mapping.address.get();
                let Some(mapping_end) = mapping_start.checked_add(mapping.size) else {
                    return false;
                };
                // Gen5 CPU_READ_WRITE is bit 1; CPU_WRITE is bit 5. GPU-only mappings must not
                // become host I/O destinations merely because their pages are readable.
                if mapping_start > cursor
                    || mapping_end <= cursor
                    || mapping.protection & (0x02 | 0x20) == 0
                {
                    return false;
                }
                cursor = mapping_end.min(end);
            }
            return true;
        }
    }
    // Unit tests and process-image allocations are not owned by KernelMemory. Their host mapping
    // is the only authoritative range information available here; watched writable pages may be
    // host-read-only, so deliberately test readability rather than attempting a write.
    process_guest_range_has_read_access(address, length)
}

pub(crate) fn ampr_command_buffer_constructor(
    arguments: [u64; 6],
    state: &mut ProbeDispatchState,
) -> u64 {
    let command_buffer = arguments[0];
    if command_buffer == 0 {
        return 0;
    }
    guest_memset(command_buffer, 0, 0x28);
    write_guest_u64(command_buffer, command_buffer);
    write_guest_u64(command_buffer + 8, arguments[1]);
    write_guest_u64(command_buffer + 0x10, arguments[2]);
    state.ampr_command_buffers.insert(
        command_buffer,
        AmprCommandBufferState {
            buffer: arguments[1],
            size: arguments[2],
            ..AmprCommandBufferState::default()
        },
    );
    command_buffer
}

pub(crate) fn ampr_apr_command_buffer_constructor(
    arguments: [u64; 6],
    state: &mut ProbeDispatchState,
) -> u64 {
    let command_buffer = arguments[0];
    if command_buffer == 0 {
        return 0;
    }
    let buffer = read_guest_u64(command_buffer + 8);
    let size = read_guest_u64(command_buffer + 0x10);
    write_guest_u64(command_buffer, command_buffer);
    write_guest_u64(command_buffer + 0x18, arguments[1]);
    write_guest_u64(command_buffer + 0x20, arguments[2]);
    state.ampr_command_buffers.insert(
        command_buffer,
        AmprCommandBufferState {
            buffer,
            size,
            ..AmprCommandBufferState::default()
        },
    );
    command_buffer
}

pub(crate) fn ampr_command_buffer_set_buffer(
    arguments: [u64; 6],
    state: &mut ProbeDispatchState,
) -> u64 {
    let command_buffer = arguments[0];
    if command_buffer == 0 {
        return sce_kernel_error(SCE_KERNEL_ERROR_INVALID_ARGUMENT);
    }
    write_guest_u64(command_buffer, command_buffer);
    write_guest_u64(command_buffer + 8, arguments[1]);
    write_guest_u64(command_buffer + 0x10, arguments[2]);
    state.ampr_command_buffers.insert(
        command_buffer,
        AmprCommandBufferState {
            buffer: arguments[1],
            size: arguments[2],
            ..AmprCommandBufferState::default()
        },
    );
    0
}

/// Identifier reported for a path the kernel could not resolve.
pub(crate) const AMPR_INVALID_FILE_ID: u32 = 0xffff_ffff;

/// Byte size of one recorded APR command record.
pub(crate) const AMPR_COMMAND_RECORD_SIZE: u64 = 0x30;
/// Record tag for an equeue completion signal.
const AMPR_COMMAND_WRITE_EVENT_QUEUE: u32 = 2;
/// Record tag for a completion word written to guest memory.
const AMPR_COMMAND_WRITE_ADDRESS: u32 = 3;

/// Reports the capacity of the memory attached to a command buffer.
pub(crate) fn ampr_command_buffer_get_size(arguments: [u64; 6], state: &ProbeDispatchState) -> u64 {
    let command_buffer = arguments[0];
    if command_buffer == 0 {
        return 0;
    }
    state
        .ampr_command_buffers
        .get(&command_buffer)
        .map_or_else(|| read_guest_u64(command_buffer + 0x10), |entry| entry.size)
}

/// Reports how many bytes of the attached memory already hold recorded commands.
pub(crate) fn ampr_command_buffer_get_current_offset(
    arguments: [u64; 6],
    state: &ProbeDispatchState,
) -> u64 {
    let command_buffer = arguments[0];
    if command_buffer == 0 {
        return 0;
    }
    state
        .ampr_command_buffers
        .get(&command_buffer)
        .map_or(0, |entry| entry.write_offset)
}

/// Reports the bytes one command occupies, so the guest can size its buffer before recording.
pub(crate) const fn ampr_measure_command_size() -> u64 {
    AMPR_COMMAND_RECORD_SIZE
}

/// Records a command that signals a kernel event queue when the buffer's work completes.
///
/// REmu executes APR transfers as they are recorded, so the queued completion is delivered when
/// the buffer is submitted rather than by a hardware pipeline.
pub(crate) fn ampr_command_buffer_write_kernel_event_queue(
    arguments: [u64; 6],
    state: &mut ProbeDispatchState,
) -> u64 {
    let command_buffer = arguments[0];
    let equeue = arguments[1];
    let ident = arguments[2];
    let user_data = arguments[3];
    if command_buffer == 0 {
        return sce_kernel_error(SCE_KERNEL_ERROR_INVALID_ARGUMENT);
    }
    let Some(command) = state.ampr_command_buffers.get_mut(&command_buffer) else {
        return sce_kernel_error(SCE_KERNEL_ERROR_INVALID_ARGUMENT);
    };
    if command.write_offset > command.size
        || AMPR_COMMAND_RECORD_SIZE > command.size - command.write_offset
    {
        return sce_kernel_error(SCE_KERNEL_ERROR_NO_SPACE);
    }
    let record = command.buffer + command.write_offset;
    guest_memset(record, 0, AMPR_COMMAND_RECORD_SIZE);
    write_guest_u32(record, AMPR_COMMAND_WRITE_EVENT_QUEUE);
    write_guest_u64(record + 8, equeue);
    write_guest_u64(record + 0x10, ident);
    write_guest_u64(record + 0x18, user_data);
    command.write_offset += AMPR_COMMAND_RECORD_SIZE;
    command.command_count += 1;
    // The completion belongs to the submission, not to recording it. Retain it so submitting the
    // buffer announces it in the order the guest recorded.
    command.pending_completions.push(AmprCompletion {
        equeue,
        ident,
        user_data,
    });
    if trace_ampr_reads() {
        eprintln!(
            "[REMu][AMPR] record completion buffer={command_buffer:#x} equeue={equeue:#x} ident={ident:#x} data={user_data:#x}"
        );
    }
    0
}

/// Records a command that writes `value` to `address` once the buffer's work completes.
///
/// This is the polled half of the completion contract: a guest that does not register an event
/// queue watches this word instead, so the write has to happen on submission, after the transfers,
/// and has to repeat on every reuse of the program.
pub(crate) fn ampr_command_buffer_write_address_on_completion(
    arguments: [u64; 6],
    state: &mut ProbeDispatchState,
) -> u64 {
    let command_buffer = arguments[0];
    let address = arguments[1];
    let value = arguments[2];
    // A null destination would make every later submission write to address zero.
    if command_buffer == 0 || address == 0 {
        return sce_kernel_error(SCE_KERNEL_ERROR_INVALID_ARGUMENT);
    }
    let Some(command) = state.ampr_command_buffers.get_mut(&command_buffer) else {
        return sce_kernel_error(SCE_KERNEL_ERROR_INVALID_ARGUMENT);
    };
    if command.write_offset > command.size
        || AMPR_COMMAND_RECORD_SIZE > command.size - command.write_offset
    {
        return sce_kernel_error(SCE_KERNEL_ERROR_NO_SPACE);
    }
    let record = command.buffer + command.write_offset;
    guest_memset(record, 0, AMPR_COMMAND_RECORD_SIZE);
    write_guest_u32(record, AMPR_COMMAND_WRITE_ADDRESS);
    write_guest_u64(record + 8, address);
    write_guest_u64(record + 0x10, value);
    command.write_offset += AMPR_COMMAND_RECORD_SIZE;
    command.command_count += 1;
    command
        .pending_address_writes
        .push(AmprAddressWrite { address, value });
    if trace_ampr_reads() {
        eprintln!(
            "[REMu][AMPR] record completion word buffer={command_buffer:#x} address={address:#x} value={value:#x}"
        );
    }
    0
}

/// Releases a command buffer's recorded state without freeing the guest's memory.
pub(crate) fn ampr_command_buffer_destructor(
    arguments: [u64; 6],
    state: &mut ProbeDispatchState,
) -> u64 {
    let command_buffer = arguments[0];
    if command_buffer == 0 {
        return sce_kernel_error(SCE_KERNEL_ERROR_INVALID_ARGUMENT);
    }
    state.ampr_command_buffers.remove(&command_buffer);
    0
}

/// Detaches a command buffer's backing memory and returns it to the caller.
///
/// This returns the buffer *pointer*, not a status code: the caller owns that allocation again
/// and typically frees it. Returning an error code here makes the guest allocator free a
/// non-pointer and abort.
pub(crate) fn ampr_command_buffer_clear_buffer(
    arguments: [u64; 6],
    state: &mut ProbeDispatchState,
) -> u64 {
    let command_buffer = arguments[0];
    if command_buffer == 0 {
        return 0;
    }
    let buffer = state
        .ampr_command_buffers
        .get(&command_buffer)
        .map_or_else(|| read_guest_u64(command_buffer + 8), |entry| entry.buffer);
    let size = state
        .ampr_command_buffers
        .get(&command_buffer)
        .map_or_else(|| read_guest_u64(command_buffer + 0x10), |entry| entry.size);
    if buffer == 0 || size == 0 {
        return 0;
    }
    write_guest_u64(command_buffer + 8, 0);
    write_guest_u64(command_buffer + 0x10, 0);
    state.ampr_command_buffers.remove(&command_buffer);
    buffer
}

pub(crate) fn ampr_command_buffer_reset(
    arguments: [u64; 6],
    state: &mut ProbeDispatchState,
) -> u64 {
    let command_buffer = arguments[0];
    if command_buffer == 0 {
        return sce_kernel_error(SCE_KERNEL_ERROR_INVALID_ARGUMENT);
    }
    let entry = state
        .ampr_command_buffers
        .entry(command_buffer)
        .or_insert_with(|| AmprCommandBufferState {
            buffer: read_guest_u64(command_buffer + 8),
            size: read_guest_u64(command_buffer + 0x10),
            ..AmprCommandBufferState::default()
        });
    entry.write_offset = 0;
    entry.command_count = 0;
    entry.pending_reads.clear();
    entry.pending_completions.clear();
    entry.pending_address_writes.clear();
    0
}

fn ampr_file_id(path: &str) -> u32 {
    path.as_bytes()
        .iter()
        .fold(2_166_136_261_u32, |hash, byte| {
            (hash ^ u32::from(*byte)).wrapping_mul(16_777_619)
        })
}

/// Optional outputs and inputs shared by every APR path-resolution entry point.
///
/// The console exposes one resolver through eight names differing only in whether the caller
/// wants sizes, supplies a path prefix, or collects a per-entry result instead of stopping at the
/// first rejection.
#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct AprResolveOutputs {
    pub(crate) ids: u64,
    pub(crate) sizes: u64,
    pub(crate) error_index: u64,
    pub(crate) results: u64,
}

pub(crate) fn ampr_resolve_file_paths(arguments: [u64; 6], state: &mut ProbeDispatchState) -> u64 {
    ampr_resolve_paths_core(
        arguments[0],
        arguments[1],
        None,
        AprResolveOutputs {
            ids: arguments[2],
            sizes: arguments[3],
            error_index: arguments[4],
            ..AprResolveOutputs::default()
        },
        state,
    )
}

/// Resolves a batch of guest paths to APR file identifiers.
///
/// Every entry is attempted when the caller supplies a per-entry result array; otherwise the
/// batch stops at the first rejection, as the console does. Rejected entries always receive the
/// invalid identifier and a zero size so a caller cannot mistake stale memory for a usable id.
pub(crate) fn ampr_resolve_paths_core(
    path_list: u64,
    count: u64,
    prefix: Option<String>,
    outputs: AprResolveOutputs,
    state: &mut ProbeDispatchState,
) -> u64 {
    let AprResolveOutputs {
        ids,
        sizes,
        error_index,
        results,
    } = outputs;
    if path_list == 0 || count == 0 || count > 1024 || (ids == 0 && sizes == 0 && results == 0) {
        return sce_kernel_error(SCE_KERNEL_ERROR_INVALID_ARGUMENT);
    }
    for index in 0..count {
        if ids != 0 {
            write_guest_u32(ids + index * 4, AMPR_INVALID_FILE_ID);
        }
        if sizes != 0 {
            write_guest_u64(sizes + index * 8, 0);
        }
    }
    let mut first_error = 0;
    let mut resolved = 0_u32;
    for index in 0..count {
        let candidate = read_guest_u64(path_list + index * 8);
        let path = read_guest_cstring(candidate).or_else(|| {
            (index == 0)
                .then(|| read_guest_cstring(path_list))
                .flatten()
        });
        let entry_error = match path {
            None => sce_kernel_error(SCE_KERNEL_ERROR_MEMORY_FAULT),
            Some(guest_path) => {
                let guest_path = match prefix.as_deref() {
                    Some(prefix) => join_guest_path(prefix, &guest_path),
                    None => guest_path,
                };
                ampr_resolve_one_path(&guest_path, index, count, ids, sizes, state)
            }
        };
        if results != 0 {
            write_guest_u32(results + index * 4, low_u32(entry_error));
        }
        if entry_error == 0 {
            resolved = resolved.saturating_add(1);
            continue;
        }
        if first_error == 0 {
            first_error = entry_error;
            if error_index != 0 {
                write_guest_u32(error_index, low_u32(index));
            }
        }
        if results == 0 {
            return entry_error;
        }
    }
    if results != 0 {
        return u64::from(resolved);
    }
    first_error
}

/// Joins a resolver prefix with one entry, keeping exactly one separator between them.
pub(crate) fn join_guest_path(prefix: &str, path: &str) -> String {
    if prefix.is_empty() {
        return path.to_owned();
    }
    let prefix = prefix.trim_end_matches('/');
    let path = path.trim_start_matches('/');
    format!("{prefix}/{path}")
}

/// Resolves one path, recording its identifier and size on success.
fn ampr_resolve_one_path(
    guest_path: &str,
    index: u64,
    count: u64,
    ids: u64,
    sizes: u64,
    state: &mut ProbeDispatchState,
) -> u64 {
    let Some((host_path, _)) = resolve_probe_guest_path(state, guest_path) else {
        eprintln!("[REMu][AMPR] resolve rejected guest={guest_path:?} index={index} of {count}");
        return sce_kernel_error(SCE_KERNEL_ERROR_NOT_FOUND);
    };
    let Ok(metadata) = host_path.metadata() else {
        eprintln!(
            "[REMu][AMPR] resolve missing guest={guest_path:?} host={:?} index={index} of {count}",
            host_path.display()
        );
        return sce_kernel_error(SCE_KERNEL_ERROR_NOT_FOUND);
    };
    let file_id = ampr_file_id(guest_path);
    state.ampr_files.insert(file_id, host_path.clone());
    state.ampr_file_handles.remove(&file_id);
    if ids != 0 {
        write_guest_u32(ids + index * 4, file_id);
    }
    if sizes != 0 {
        write_guest_u64(sizes + index * 8, metadata.len());
    }
    // Success is the common case by orders of magnitude: a title that indexes its whole asset
    // database resolves hundreds of thousands of paths during startup, and logging each one cost
    // more than the resolution itself. Failures above stay unconditional, because they are rare
    // and are what a bring-up needs to see.
    if trace_ampr_reads() {
        eprintln!(
            "[REMu][AMPR] resolved guest={guest_path:?} host={:?} id={file_id:#x} size={:#x}",
            host_path.display(),
            metadata.len()
        );
    }
    0
}

/// Reports the size of an already resolved APR file.
pub(crate) fn ampr_get_file_size(arguments: [u64; 6], state: &ProbeDispatchState) -> u64 {
    let (file_id, output) = (low_u32(arguments[0]), arguments[1]);
    if output == 0 {
        return sce_kernel_error(SCE_KERNEL_ERROR_INVALID_ARGUMENT);
    }
    let Some(host_path) = state.ampr_files.get(&file_id) else {
        return sce_kernel_error(SCE_KERNEL_ERROR_NOT_FOUND);
    };
    let Ok(metadata) = host_path.metadata() else {
        return sce_kernel_error(SCE_KERNEL_ERROR_NOT_FOUND);
    };
    write_guest_u64(output, metadata.len());
    0
}

/// Fills a kernel stat structure for an already resolved APR file.
pub(crate) fn ampr_get_file_stat(arguments: [u64; 6], state: &ProbeDispatchState) -> u64 {
    let (file_id, output) = (low_u32(arguments[0]), arguments[1]);
    if output == 0 {
        return sce_kernel_error(SCE_KERNEL_ERROR_INVALID_ARGUMENT);
    }
    let Some(host_path) = state.ampr_files.get(&file_id).cloned() else {
        return sce_kernel_error(SCE_KERNEL_ERROR_NOT_FOUND);
    };
    let Ok(metadata) = host_path.metadata() else {
        return sce_kernel_error(SCE_KERNEL_ERROR_NOT_FOUND);
    };
    let payload = kernel_stat_payload_from_metadata(&metadata, &file_id.to_le_bytes());
    // SAFETY: the caller supplies a writable Gen5 stat object in live guest memory.
    unsafe {
        std::ptr::copy_nonoverlapping(payload.as_ptr(), output as *mut u8, payload.len());
    }
    0
}

pub(crate) fn ampr_command_buffer_read_file_ordered(
    arguments: [u64; 6],
    stack_arguments: [u64; 6],
) -> u64 {
    let command_buffer = arguments[0];
    let order = ampr_command_order(command_buffer);
    let _order = order
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let dispatch = PROBE_DISPATCH.get_or_init(|| RwLock::new(ProbeDispatchState::default()));
    let mut state = dispatch
        .write()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    ampr_command_buffer_record_read(arguments, stack_arguments, &mut state)
}

pub(crate) fn ampr_command_buffer_record_read(
    arguments: [u64; 6],
    stack_arguments: [u64; 6],
    state: &mut ProbeDispatchState,
) -> u64 {
    let command_buffer = arguments[0];
    let file_id = low_u32(arguments[3]);
    let destination = arguments[4];
    let size = arguments[5];
    let file_offset = stack_arguments[0];
    if command_buffer == 0 || (destination == 0 && size != 0) {
        return sce_kernel_error(SCE_KERNEL_ERROR_INVALID_ARGUMENT);
    }
    let Ok(capacity) = usize::try_from(size) else {
        return sce_kernel_error(SCE_KERNEL_ERROR_INVALID_ARGUMENT);
    };
    if !state.ampr_files.contains_key(&file_id) {
        eprintln!(
            "[REMu][AMPR] read rejected unknown file id={file_id:#x} args={:#x},{:#x},{:#x},{:#x},{:#x},{:#x} stack={:#x},{:#x} known=[{}]",
            arguments[0],
            arguments[1],
            arguments[2],
            arguments[3],
            arguments[4],
            arguments[5],
            stack_arguments[0],
            stack_arguments[1],
            state
                .ampr_files
                .keys()
                .map(|id| format!("{id:#x}"))
                .collect::<Vec<_>>()
                .join(",")
        );
        return sce_kernel_error(SCE_KERNEL_ERROR_NOT_FOUND);
    }
    if !ampr_destination_has_write_access(state, destination, capacity) {
        eprintln!(
            "[REMu][AMPR] read rejected unmapped destination={destination:#x} size={size:#x}"
        );
        return sce_kernel_error(SCE_KERNEL_ERROR_MEMORY_FAULT);
    }
    let Some(command) = state.ampr_command_buffers.get_mut(&command_buffer) else {
        return sce_kernel_error(SCE_KERNEL_ERROR_INVALID_ARGUMENT);
    };
    if command.write_offset > command.size
        || AMPR_COMMAND_RECORD_SIZE > command.size - command.write_offset
    {
        return sce_kernel_error(SCE_KERNEL_ERROR_INVALID_ARGUMENT);
    }
    let record = command.buffer + command.write_offset;
    guest_memset(record, 0, AMPR_COMMAND_RECORD_SIZE);
    write_guest_u32(record, 1);
    write_guest_u32(record + 4, file_id);
    write_guest_u64(record + 8, destination);
    write_guest_u64(record + 0x10, size);
    write_guest_u64(record + 0x18, file_offset);
    // The transfer has not run yet. Submission overwrites this diagnostic/result field with the
    // number of bytes actually copied.
    write_guest_u64(record + 0x20, 0);
    command.pending_reads.push(AmprReadCommand {
        file_id,
        destination,
        size,
        file_offset,
        record_offset: command.write_offset,
    });
    command.write_offset += AMPR_COMMAND_RECORD_SIZE;
    command.command_count += 1;
    if trace_ampr_reads() {
        eprintln!(
            "[REMu][AMPR] record read id={file_id:#x} offset={file_offset:#x} requested={size:#x} destination={destination:#x} thread={:?}",
            current_guest_thread_name()
        );
    }
    0
}

pub(crate) fn ampr_read_file_to_guest(
    file: &std::fs::File,
    file_id: u32,
    destination: u64,
    size: u64,
    file_offset: u64,
    capacity: usize,
) -> Result<usize, u64> {
    if capacity == 0 {
        return Ok(0);
    }
    if !process_guest_range_has_read_access(destination, capacity) {
        eprintln!(
            "[REMu][AMPR] read rejected unmapped destination={destination:#x} size={size:#x}"
        );
        return Err(sce_kernel_error(SCE_KERNEL_ERROR_MEMORY_FAULT));
    }
    // A watched guest page is host read-only. macOS pread(2) returns EFAULT instead of delivering
    // the write-fault signal, and a concurrent resource upload can rearm the page between an eager
    // mprotect and the syscall. Read into ordinary host storage, then copy with CPU stores so the
    // write-watch handler dirties each page synchronously.
    let mut staging = AMPR_READ_STAGING_POOL
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .pop()
        .unwrap_or_default();
    let additional = capacity.saturating_sub(staging.len());
    if let Err(error) = staging.try_reserve(additional) {
        eprintln!("[REMu][AMPR] could not stage read id={file_id:#x} size={size:#x}: {error}");
        return Err(sce_kernel_error(SCE_KERNEL_ERROR_OUT_OF_MEMORY));
    }
    staging.resize(capacity, 0_u8);
    let mut total = 0_usize;
    while total < staging.len() {
        let Some(offset) = file_offset.checked_add(total as u64) else {
            return Err(sce_kernel_error(SCE_KERNEL_ERROR_INVALID_ARGUMENT));
        };
        let read = match positional_file_read(file, &mut staging[total..], offset) {
            Ok(file) => file,
            Err(error) => {
                eprintln!(
                    "[REMu][AMPR] read failed id={file_id:#x} offset={offset:#x} \
remaining={:#x} destination={:#x}: {error}",
                    staging.len() - total,
                    destination.saturating_add(total as u64)
                );
                return Err(sce_kernel_error(SCE_KERNEL_ERROR_NOT_FOUND));
            }
        };
        if read == 0 {
            break;
        }
        total += read;
    }
    // SAFETY: the complete guest destination range was validated above. Ordinary CPU stores enter
    // CpuWriteWatch's signal handler on protected pages before changing their bytes.
    unsafe {
        std::ptr::copy_nonoverlapping(staging.as_ptr(), destination as *mut u8, total);
    }
    if staging.capacity() <= AMPR_MAX_RETAINED_STAGING_BYTES {
        staging.clear();
        let mut pool = AMPR_READ_STAGING_POOL
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        if pool.len() < AMPR_STAGING_POOL_LIMIT {
            pool.push(staging);
        }
    }
    Ok(total)
}

pub(crate) fn ampr_submit_command_buffer_ordered(arguments: [u64; 6]) -> u64 {
    let order = ampr_command_order(arguments[0]);
    let dispatch = PROBE_DISPATCH.get_or_init(|| RwLock::new(ProbeDispatchState::default()));
    ampr_submit_command_buffer_two_phase_with(order, dispatch, arguments, ampr_read_file_to_guest)
}

pub(crate) fn ampr_submit_command_buffer_two_phase_with<F>(
    order: &Mutex<()>,
    dispatch: &RwLock<ProbeDispatchState>,
    arguments: [u64; 6],
    mut read_file: F,
) -> u64
where
    F: FnMut(&std::fs::File, u32, u64, u64, u64, usize) -> Result<usize, u64>,
{
    let _order = order
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let command_buffer = arguments[0];
    let (reads, paths) = {
        let state = dispatch
            .read()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let Some(command) = state.ampr_command_buffers.get(&command_buffer) else {
            return sce_kernel_error(SCE_KERNEL_ERROR_INVALID_ARGUMENT);
        };
        let mut paths = BTreeMap::new();
        for read in &command.pending_reads {
            let Ok(capacity) = usize::try_from(read.size) else {
                return sce_kernel_error(SCE_KERNEL_ERROR_INVALID_ARGUMENT);
            };
            // A reusable APR program may outlive or reprotect its original destination. Validate
            // the entire current mapping on every submission rather than trusting record time.
            if !ampr_destination_has_write_access(&state, read.destination, capacity) {
                return sce_kernel_error(SCE_KERNEL_ERROR_MEMORY_FAULT);
            }
            let Some(path) = state.ampr_files.get(&read.file_id).cloned() else {
                return sce_kernel_error(SCE_KERNEL_ERROR_NOT_FOUND);
            };
            paths
                .entry(read.file_id)
                .or_insert_with(|| (path, state.ampr_file_handles.get(&read.file_id).cloned()));
        }
        (command.pending_reads.clone(), paths)
    };

    let mut files = BTreeMap::<u32, Arc<std::fs::File>>::new();
    for (file_id, (path, cached)) in paths {
        let file = match cached {
            Some(file) => file,
            None => match std::fs::File::open(&path) {
                Ok(file) => Arc::new(file),
                Err(error) => {
                    eprintln!(
                        "[REMu][AMPR] submit could not open id={file_id:#x} host={:?}: {error}",
                        path.display()
                    );
                    return sce_kernel_error(SCE_KERNEL_ERROR_NOT_FOUND);
                }
            },
        };
        files.insert(file_id, file);
    }

    // Execute the reusable program in record order. Neither host storage latency nor watched
    // guest-memory faults retain the process dispatch lock, so unrelated guest services and APR
    // buffers continue making progress.
    let mut completed_reads = Vec::with_capacity(reads.len());
    for read in reads {
        let Ok(capacity) = usize::try_from(read.size) else {
            return sce_kernel_error(SCE_KERNEL_ERROR_INVALID_ARGUMENT);
        };
        {
            let state = dispatch
                .read()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            // Opening another file, or an earlier command, may have allowed the guest to change
            // this reusable destination. Revalidate directly before this command's I/O.
            if !ampr_destination_has_write_access(&state, read.destination, capacity) {
                return sce_kernel_error(SCE_KERNEL_ERROR_MEMORY_FAULT);
            }
        }
        let Some(file) = files.get(&read.file_id) else {
            return sce_kernel_error(SCE_KERNEL_ERROR_NOT_FOUND);
        };
        let bytes_read = match read_file(
            file,
            read.file_id,
            read.destination,
            read.size,
            read.file_offset,
            capacity,
        ) {
            Ok(bytes_read) => bytes_read,
            Err(error) => return error,
        };
        completed_reads.push((read.record_offset, bytes_read));
    }

    let mut state = dispatch
        .write()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    for (file_id, file) in files {
        state.ampr_file_handles.entry(file_id).or_insert(file);
    }
    let Some(command) = state.ampr_command_buffers.get(&command_buffer) else {
        return sce_kernel_error(SCE_KERNEL_ERROR_INVALID_ARGUMENT);
    };
    for (record_offset, bytes_read) in completed_reads {
        let Some(result_offset) = record_offset.checked_add(0x20) else {
            return sce_kernel_error(SCE_KERNEL_ERROR_INVALID_ARGUMENT);
        };
        if result_offset > command.size || 8 > command.size - result_offset {
            return sce_kernel_error(SCE_KERNEL_ERROR_INVALID_ARGUMENT);
        }
        write_guest_u64(
            command.buffer + result_offset,
            u64::try_from(bytes_read).unwrap_or(u64::MAX),
        );
    }
    ampr_submit_command_buffer(arguments, &mut state)
}

pub(crate) fn ampr_submit_command_buffer(
    arguments: [u64; 6],
    state: &mut ProbeDispatchState,
) -> u64 {
    if arguments[0] == 0 || !state.ampr_command_buffers.contains_key(&arguments[0]) {
        return sce_kernel_error(SCE_KERNEL_ERROR_INVALID_ARGUMENT);
    }
    state.next_apr_submission_id = state.next_apr_submission_id.wrapping_add(1).max(1);
    let id = state.next_apr_submission_id;
    state.apr_submissions.insert(id);
    // A command buffer is a reusable program. Every submission repeats its completion commands
    // after the transfer pass above, until the guest resets or destroys the program.
    let completions = state
        .ampr_command_buffers
        .get(&arguments[0])
        .map(|command| command.pending_completions.clone())
        .unwrap_or_default();
    let address_writes = state
        .ampr_command_buffers
        .get(&arguments[0])
        .map(|command| command.pending_address_writes.clone())
        .unwrap_or_default();
    let announced = !completions.is_empty();
    if trace_ampr_reads() {
        eprintln!(
            "[REMu][AMPR] submit buffer={:#x} id={id} completions={} words={}",
            arguments[0],
            completions.len(),
            address_writes.len()
        );
    }
    // Publish the polled completion words before announcing any event, so a guest woken by the
    // queue already observes the word a peer thread may be spinning on.
    for write in address_writes {
        if !ampr_destination_has_write_access(state, write.address, 8) {
            return sce_kernel_error(SCE_KERNEL_ERROR_MEMORY_FAULT);
        }
        write_guest_u64(write.address, write.value);
    }
    // The value recorded with the queue-write command is the event's `data`, which the guest
    // reads back through `sceKernelGetEventData` to identify which request completed. `udata`
    // belongs to the registration, not to the individual command.
    for completion in completions {
        let user_data = state
            .kernel_event_registrations
            .get(&(
                completion.equeue,
                KERNEL_EVENT_FILTER_AMPR,
                completion.ident,
            ))
            .copied()
            .unwrap_or(0);
        queue_kernel_event(
            state,
            KernelQueuedEvent {
                equeue: completion.equeue,
                ident: completion.ident,
                filter: KERNEL_EVENT_FILTER_AMPR,
                flags: 0,
                fflags: 1,
                data: completion.user_data,
                user_data,
                agc_graphics_completions: 0,
                agc_user_interrupts: 0,
            },
        );
    }
    if announced {
        notify_kernel_equeue_waiters();
    }
    if arguments[2] != 0 {
        write_guest_u64(arguments[2], 0);
    }
    if arguments[3] != 0 {
        write_guest_u32(arguments[3], id);
    }
    0
}

pub(crate) fn ampr_wait_command_buffer(arguments: [u64; 6], state: &mut ProbeDispatchState) -> u64 {
    let id = low_u32(arguments[0]);
    let known = state.apr_submissions.remove(&id);
    if trace_ampr_reads() {
        eprintln!("[REMu][AMPR] wait id={id} known={known}");
    }
    if known {
        0
    } else {
        sce_kernel_error(SCE_KERNEL_ERROR_NOT_FOUND)
    }
}
