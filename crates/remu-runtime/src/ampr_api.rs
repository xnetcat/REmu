use std::collections::BTreeMap;
#[cfg(unix)]
use std::os::unix::fs::FileExt;
#[cfg(windows)]
use std::os::windows::fs::FileExt;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, RwLock};

use remu_shader::GuestMemoryReader;

use crate::{
    AMPR_COMMAND_ORDER, AMPR_READ_STAGING_POOL, AmprAddressWrite, AmprCommandBufferState,
    AmprCompletion, AmprReadCommand, KERNEL_EVENT_FILTER_AMPR, KernelQueuedEvent, PROBE_DISPATCH,
    ProbeDispatchState, ProcessGuestMemory, SCE_KERNEL_ERROR_INVALID_ARGUMENT,
    SCE_KERNEL_ERROR_MEMORY_FAULT, SCE_KERNEL_ERROR_NO_SPACE, SCE_KERNEL_ERROR_NOT_FOUND,
    SCE_KERNEL_ERROR_OUT_OF_MEMORY, current_guest_thread_name, guest_memset,
    kernel_stat_payload_from_metadata, lock_probe_dispatch_for_write, low_u32,
    notify_kernel_equeue_waiters, process_guest_range_has_read_access, queue_kernel_event,
    read_guest_u64, resolve_probe_guest_path_with_presence, sce_kernel_error, trace_ampr_reads,
    try_read_process_guest_u64, write_guest_u32, write_guest_u64,
};

const AMPR_ORDER_LOCK_COUNT: usize = 64;
const AMPR_STAGING_POOL_LIMIT: usize = 4;
const AMPR_MAX_RETAINED_STAGING_BYTES: usize = 8 * 1024 * 1024;
const AMPR_MAX_PATH_BYTES: usize = 1024;
/// Published to a recorded completion word when the submission's transfers did not all run.
///
/// Zero is the word's pre-submission state, so a failure cannot be reported by leaving it or by
/// clearing it: a guest polling the word would never observe the submission ending. The value is
/// deliberately not the recorded success token, which means "the transfers this program records
/// have happened".
const AMPR_FAILED_COMPLETION_VALUE: u64 = u64::MAX;
/// `EV_ERROR`. Set on a completion event whose transfers failed; `data` still carries the
/// recorded token, so the guest can identify which request the error belongs to.
const AMPR_EVENT_FLAG_ERROR: u16 = 0x4000;

fn positional_file_read_once(
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

/// Reads with a bounded retry on transient I/O errors.
///
/// Removable/USB-bridge volumes intermittently fail a positional read with EIO and recover on
/// the next attempt; surfacing the first failure turns into a guest LowLevelFatalError ("Apr
/// read failure 80020002") that kills the whole title. Genuine persistent failures still
/// surface after the last attempt. Retries stay short and bounded so a truly dead volume
/// costs under half a second per read rather than hanging.
fn positional_file_read(
    file: &std::fs::File,
    destination: &mut [u8],
    offset: u64,
) -> std::io::Result<usize> {
    remu_memory::debug_log!(
        "ampr",
        "read offset={offset:#x} bytes={:#x}",
        destination.len()
    );
    let mut delay = std::time::Duration::from_millis(20);
    for attempt in 0..5_u32 {
        match positional_file_read_once(file, destination, offset) {
            Ok(read) => {
                if attempt != 0 {
                    eprintln!(
                        "[REMu][AMPR] transient read failure recovered after {attempt} retries offset={offset:#x} bytes={}",
                        destination.len(),
                    );
                }
                return Ok(read);
            }
            Err(error)
                if attempt < 4
                    && matches!(
                        error.kind(),
                        std::io::ErrorKind::Interrupted | std::io::ErrorKind::TimedOut
                    )
                    || attempt < 4 && error.raw_os_error() == Some(5) =>
            {
                std::thread::sleep(delay);
                delay *= 2;
            }
            Err(error) => return Err(error),
        }
    }
    unreachable!("bounded retry loop returns on the final attempt")
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

/// Upper bound on APR descriptors held open simultaneously (G1).
///
/// Sits well under the raised host descriptor budget (`DESCRIPTOR_CEILING = 10240` in `lib.rs`) so
/// the guest's own open files, sockets, and the emulator's host resources keep their headroom while
/// the APR cache still keeps the working set of recently-read assets warm.
pub(crate) const AMPR_MAX_OPEN_HANDLES: usize = 2048;

impl ProbeDispatchState {
    /// Returns the stable slot identifier for a resolved host path, allocating one on first use.
    ///
    /// Identifiers are dense and monotonic from one, so distinct paths never share an identifier —
    /// the collision the former FNV-32-of-path scheme could produce (serving a different file's
    /// bytes through a colliding id) is impossible by construction. `None` is returned only if the
    /// 32-bit identifier space is exhausted, which a single process cannot reach in practice.
    pub(crate) fn intern_ampr_file_id(&mut self, host_path: &Path) -> Option<u32> {
        if let Some(id) = self.ampr_file_ids.get(host_path) {
            return Some(*id);
        }
        let id = self.ampr_next_file_id.checked_add(1)?;
        if id == AMPR_INVALID_FILE_ID {
            return None;
        }
        self.ampr_next_file_id = id;
        self.ampr_file_ids.insert(host_path.to_owned(), id);
        Some(id)
    }

    /// Retains an open APR descriptor, evicting the least-recently-used one past the budget.
    fn retain_ampr_file_handle(&mut self, file_id: u32, file: Arc<std::fs::File>) {
        self.retain_ampr_file_handle_within(file_id, file, AMPR_MAX_OPEN_HANDLES);
    }

    /// Retains a descriptor under an explicit budget so the eviction policy is unit-testable.
    fn retain_ampr_file_handle_within(
        &mut self,
        file_id: u32,
        file: Arc<std::fs::File>,
        capacity: usize,
    ) {
        self.ampr_handle_order.retain(|id| *id != file_id);
        self.ampr_file_handles.insert(file_id, file);
        self.ampr_handle_order.push_back(file_id);
        while self.ampr_file_handles.len() > capacity.max(1) {
            let Some(evicted) = self.ampr_handle_order.pop_front() else {
                break;
            };
            self.ampr_file_handles.remove(&evicted);
        }
    }

    /// Drops a descriptor and its recency record together so no stale identifier lingers in the
    /// eviction order after a path remapping.
    fn forget_ampr_file_handle(&mut self, file_id: u32) {
        self.ampr_file_handles.remove(&file_id);
        self.ampr_handle_order.retain(|id| *id != file_id);
    }
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
    if arguments[3] == 0 {
        return sce_kernel_error(SCE_KERNEL_ERROR_INVALID_ARGUMENT);
    }
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
        eprintln!(
            "[REMu][AMPR] resolve rejected arguments path_list={path_list:#x} count={count} \
             ids={ids:#x} sizes={sizes:#x} results={results:#x}"
        );
        return sce_kernel_error(SCE_KERNEL_ERROR_INVALID_ARGUMENT);
    }
    let Ok(count_usize) = usize::try_from(count) else {
        return sce_kernel_error(SCE_KERNEL_ERROR_INVALID_ARGUMENT);
    };
    let Some(path_list_bytes) = count_usize.checked_mul(std::mem::size_of::<u64>()) else {
        return sce_kernel_error(SCE_KERNEL_ERROR_INVALID_ARGUMENT);
    };
    if !process_guest_range_has_read_access(path_list, path_list_bytes)
        || !apr_output_has_write_access(state, ids, count_usize, std::mem::size_of::<u32>())
        || !apr_output_has_write_access(state, sizes, count_usize, std::mem::size_of::<u64>())
        || !apr_output_has_write_access(state, results, count_usize, std::mem::size_of::<u32>())
        || (error_index != 0 && !ampr_destination_has_write_access(state, error_index, 4))
    {
        return sce_kernel_error(SCE_KERNEL_ERROR_MEMORY_FAULT);
    }
    let mut first_error = 0;
    let mut resolved = 0_u32;
    for index in 0..count {
        // Fail-fast resolvers leave entries after the first failure untouched. Initialize only
        // the entry that is about to be attempted, so stale data cannot make that entry usable
        // while later entries retain the caller's state exactly as the batch ABI requires.
        if ids != 0 {
            write_guest_u32(ids + index * 4, AMPR_INVALID_FILE_ID);
        }
        if sizes != 0 {
            write_guest_u64(sizes + index * 8, 0);
        }
        let candidate = try_read_process_guest_u64(path_list + index * 8).unwrap_or(0);
        let path = ampr_read_guest_path(candidate).or_else(|| {
            (index == 0)
                .then(|| ampr_read_guest_path(path_list))
                .flatten()
        });
        let entry_error = match path {
            None => {
                eprintln!(
                    "[REMu][AMPR] resolve entry unreadable index={index} of {count} \
                     entry={candidate:#x} list={path_list:#x}"
                );
                sce_kernel_error(SCE_KERNEL_ERROR_MEMORY_FAULT)
            }
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

fn apr_output_has_write_access(
    state: &ProbeDispatchState,
    address: u64,
    count: usize,
    element_size: usize,
) -> bool {
    address == 0
        || count
            .checked_mul(element_size)
            .is_some_and(|bytes| ampr_destination_has_write_access(state, address, bytes))
}

/// Reads one APR path without dereferencing an untrusted guest pointer.
pub(crate) fn ampr_read_guest_path(address: u64) -> Option<String> {
    ampr_read_guest_path_with(&ProcessGuestMemory, address)
}

fn ampr_read_guest_path_with(memory: &impl GuestMemoryReader, address: u64) -> Option<String> {
    if address == 0 {
        return None;
    }
    const READ_CHUNK_BYTES: usize = 256;
    let mut bytes = Vec::with_capacity(64);
    let mut offset = 0_usize;
    let mut scalar_fallback = false;
    while offset < AMPR_MAX_PATH_BYTES {
        let remaining = AMPR_MAX_PATH_BYTES - offset;
        let current = address.checked_add(u64::try_from(offset).ok()?)?;
        let readable = memory.readable_prefix_length(current, remaining);
        if readable == Some(0) {
            return None;
        }
        let mut length = if scalar_fallback {
            1
        } else {
            readable.unwrap_or(remaining).min(READ_CHUNK_BYTES)
        };
        let mut chunk = [0_u8; READ_CHUNK_BYTES];
        if !memory.read(current, &mut chunk[..length]) {
            // Readers without a zero-copy range query can reject a bulk read which crosses the
            // end of an otherwise valid mapping. Fall back to byte reads for this uncommon case,
            // retaining the resolver's prior fault behavior without penalizing mapped paths.
            scalar_fallback = true;
            length = 1;
            if !memory.read(current, &mut chunk[..1]) {
                return None;
            }
        }
        if let Some(terminator) = chunk[..length].iter().position(|byte| *byte == 0) {
            bytes.extend_from_slice(&chunk[..terminator]);
            return String::from_utf8(bytes).ok();
        }
        bytes.extend_from_slice(&chunk[..length]);
        offset += length;
    }
    None
}

pub(crate) fn ampr_resolve_paths_with_prefix(
    prefix_address: u64,
    path_list: u64,
    count: u64,
    outputs: AprResolveOutputs,
    state: &mut ProbeDispatchState,
) -> u64 {
    let prefix = if prefix_address == 0 {
        None
    } else {
        let Some(prefix) = ampr_read_guest_path(prefix_address) else {
            return sce_kernel_error(SCE_KERNEL_ERROR_MEMORY_FAULT);
        };
        Some(prefix)
    };
    ampr_resolve_paths_core(path_list, count, prefix, outputs, state)
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
    let Some(resolved) = resolve_probe_guest_path_with_presence(state, guest_path) else {
        eprintln!("[REMu][AMPR] resolve rejected guest={guest_path:?} index={index} of {count}");
        return sce_kernel_error(SCE_KERNEL_ERROR_NOT_FOUND);
    };
    let host_path = resolved.path;
    if resolved.known_absent {
        eprintln!(
            "[REMu][AMPR] resolve absent guest={guest_path:?} host={:?} index={index} of {count}",
            host_path.display()
        );
        return sce_kernel_error(SCE_KERNEL_ERROR_NOT_FOUND);
    }
    let Ok(metadata) = host_path.metadata() else {
        eprintln!(
            "[REMu][AMPR] resolve missing guest={guest_path:?} host={:?} index={index} of {count}",
            host_path.display()
        );
        return sce_kernel_error(SCE_KERNEL_ERROR_NOT_FOUND);
    };
    let Some(file_id) = state.intern_ampr_file_id(&host_path) else {
        eprintln!(
            "[REMu][AMPR] resolve exhausted the APR identifier space at guest={guest_path:?} \
             host={:?}",
            host_path.display()
        );
        return sce_kernel_error(SCE_KERNEL_ERROR_OUT_OF_MEMORY);
    };
    ampr_record_resolved_file(state, file_id, host_path.clone());
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

/// Records the physical file behind an APR identifier without throwing away a still-valid open
/// descriptor when a title resolves the same asset again.
///
/// File identifiers are only 32 bits, so a distinct path can replace an existing mapping. That
/// case must evict the old handle or a later positional read would silently access the previous
/// file through the colliding identifier.
fn ampr_record_resolved_file(state: &mut ProbeDispatchState, file_id: u32, host_path: PathBuf) {
    let previous = state.ampr_files.insert(file_id, host_path.clone());
    if previous.as_ref() != Some(&host_path) {
        state.forget_ampr_file_handle(file_id);
    }
}

/// Reports the size of an already resolved APR file.
pub(crate) fn ampr_get_file_size(arguments: [u64; 6], state: &ProbeDispatchState) -> u64 {
    let (file_id, output) = (low_u32(arguments[0]), arguments[1]);
    if output == 0 {
        return sce_kernel_error(SCE_KERNEL_ERROR_INVALID_ARGUMENT);
    }
    let Some(host_path) = state.ampr_files.get(&file_id) else {
        eprintln!("[REMu][AMPR] get-file-size unknown id={file_id:#x}");
        return sce_kernel_error(SCE_KERNEL_ERROR_NOT_FOUND);
    };
    let Ok(metadata) = host_path.metadata() else {
        eprintln!(
            "[REMu][AMPR] get-file-size missing id={file_id:#x} host={:?}",
            host_path.display()
        );
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
        eprintln!("[REMu][AMPR] get-file-stat unknown id={file_id:#x}");
        return sce_kernel_error(SCE_KERNEL_ERROR_NOT_FOUND);
    };
    let Ok(metadata) = host_path.metadata() else {
        eprintln!(
            "[REMu][AMPR] get-file-stat missing id={file_id:#x} host={:?}",
            host_path.display()
        );
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
    let mut state = lock_probe_dispatch_for_write(dispatch);
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
    // This destination was recorded at record time and is replayed on EVERY submission of the
    // command buffer; a destination the guest has since freed is exactly the corruption class the
    // journal exists to catch.
    remu_memory::record_guest_write(
        remu_memory::GuestWriteSite::AmprReadReplay,
        destination,
        total as u64,
        &staging[..total.min(4096)],
        0,
    );
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
    let (reads, paths, retained_handles) = {
        let state = dispatch
            .read()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let Some(command) = state.ampr_command_buffers.get(&command_buffer) else {
            return sce_kernel_error(SCE_KERNEL_ERROR_INVALID_ARGUMENT);
        };
        let mut paths = BTreeMap::new();
        for read in &command.pending_reads {
            // Every per-read rejection is decided by the execution pass below, which revalidates
            // after any intervening open anyway. This pass only collects the handles that pass
            // may need, so one unresolvable record cannot cancel the whole program.
            let Some(path) = state.ampr_files.get(&read.file_id).cloned() else {
                continue;
            };
            paths
                .entry(read.file_id)
                .or_insert_with(|| (path, state.ampr_file_handles.get(&read.file_id).cloned()));
        }
        (
            command.pending_reads.clone(),
            paths,
            state.ampr_file_handles.len(),
        )
    };

    let mut files = BTreeMap::<u32, Arc<std::fs::File>>::new();
    for (file_id, (path, cached)) in paths {
        let file = match cached {
            Some(file) => file,
            None => match std::fs::File::open(&path) {
                Ok(file) => Arc::new(file),
                Err(error) => {
                    // The retained-handle count is part of the diagnostic: this cache holds one
                    // descriptor per file the title has ever read, so it is the first suspect
                    // when the host refuses an open.
                    eprintln!(
                        "[REMu][AMPR] submit could not open id={file_id:#x} host={:?} handles={retained_handles}: {error}",
                        path.display()
                    );
                    continue;
                }
            },
        };
        files.insert(file_id, file);
    }

    // Execute the reusable program in record order. Neither host storage latency nor watched
    // guest-memory faults retain the process dispatch lock, so unrelated guest services and APR
    // buffers continue making progress.
    let mut completed_reads = Vec::with_capacity(reads.len());
    let mut failure = None;
    for read in reads {
        // A failed record is contained to itself: later records still run, and its result word
        // still reports zero bytes, because a guest parses the destination on the strength of
        // that word and would otherwise parse the previous submission's bytes.
        let bytes_read = match ampr_execute_recorded_read(dispatch, &files, &read, &mut read_file) {
            Ok(bytes_read) => bytes_read,
            Err(error) => {
                failure.get_or_insert(error);
                0
            }
        };
        completed_reads.push((read.record_offset, bytes_read));
    }

    let mut state = lock_probe_dispatch_for_write(dispatch);
    for (file_id, file) in files {
        // Retain through the LRU cache so the descriptor working set stays bounded under the
        // raised fd budget; a re-read of a cached file also promotes it here.
        state.retain_ampr_file_handle(file_id, file);
    }
    let Some(command) = state.ampr_command_buffers.get(&command_buffer) else {
        return sce_kernel_error(SCE_KERNEL_ERROR_INVALID_ARGUMENT);
    };
    for (record_offset, bytes_read) in completed_reads {
        let Some(result_offset) = record_offset.checked_add(0x20) else {
            failure.get_or_insert(sce_kernel_error(SCE_KERNEL_ERROR_INVALID_ARGUMENT));
            continue;
        };
        if result_offset > command.size || 8 > command.size - result_offset {
            failure.get_or_insert(sce_kernel_error(SCE_KERNEL_ERROR_INVALID_ARGUMENT));
            continue;
        }
        let result_value = u64::try_from(bytes_read).unwrap_or(u64::MAX);
        write_guest_u64(command.buffer + result_offset, result_value);
        remu_memory::record_guest_write(
            remu_memory::GuestWriteSite::AmprCompletion,
            command.buffer + result_offset,
            8,
            &result_value.to_le_bytes(),
            0,
        );
    }
    ampr_publish_command_buffer_completion(arguments, &mut state, failure)
}

/// Runs one recorded transfer, reporting the error the guest should see when it cannot run.
///
/// Errors are per record. The caller owns the program's outcome, so nothing here may abandon the
/// remaining records or the buffer's completion publication.
fn ampr_execute_recorded_read<F>(
    dispatch: &RwLock<ProbeDispatchState>,
    files: &BTreeMap<u32, Arc<std::fs::File>>,
    read: &AmprReadCommand,
    read_file: &mut F,
) -> Result<usize, u64>
where
    F: FnMut(&std::fs::File, u32, u64, u64, u64, usize) -> Result<usize, u64>,
{
    let Ok(capacity) = usize::try_from(read.size) else {
        return Err(sce_kernel_error(SCE_KERNEL_ERROR_INVALID_ARGUMENT));
    };
    {
        let state = dispatch
            .read()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        // A reusable APR program may outlive or reprotect its original destination, and opening
        // another file or running an earlier command gives the guest the chance to. Validate the
        // entire current mapping directly before this command's I/O.
        if !ampr_destination_has_write_access(&state, read.destination, capacity) {
            eprintln!(
                "[REMu][AMPR] submit rejected unmapped destination={:#x} size={:#x} id={:#x}",
                read.destination, read.size, read.file_id
            );
            return Err(sce_kernel_error(SCE_KERNEL_ERROR_MEMORY_FAULT));
        }
    }
    let Some(file) = files.get(&read.file_id) else {
        return Err(sce_kernel_error(SCE_KERNEL_ERROR_NOT_FOUND));
    };
    read_file(
        file,
        read.file_id,
        read.destination,
        read.size,
        read.file_offset,
        capacity,
    )
}

pub(crate) fn ampr_submit_command_buffer(
    arguments: [u64; 6],
    state: &mut ProbeDispatchState,
) -> u64 {
    ampr_publish_command_buffer_completion(arguments, state, None)
}

/// Publishes a submission's recorded completions, reporting `failure` to the guest when the
/// transfers did not all run.
///
/// A failed submission publishes exactly what a successful one does. The completion word and the
/// queued event are the only channels a guest has for learning that a submission ended, so
/// withholding them turns a recoverable I/O error into a wait that never returns.
fn ampr_publish_command_buffer_completion(
    arguments: [u64; 6],
    state: &mut ProbeDispatchState,
    failure: Option<u64>,
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
        let value = if failure.is_some() {
            AMPR_FAILED_COMPLETION_VALUE
        } else {
            write.value
        };
        write_guest_u64(write.address, value);
        remu_memory::record_guest_write(
            remu_memory::GuestWriteSite::AmprCompletion,
            write.address,
            8,
            &value.to_le_bytes(),
            0,
        );
    }
    // The value recorded with the queue-write command is the event's `data`, which the guest
    // reads back through `sceKernelGetEventData` to identify which request completed. `udata`
    // belongs to the registration, not to the individual command.
    for completion in completions {
        let user_data = {
            let (_h, events) = crate::event_state::lock_event_state_for_read();
            events
                .kernel_event_registrations
                .get(&(
                    completion.equeue,
                    KERNEL_EVENT_FILTER_AMPR,
                    completion.ident,
                ))
                .copied()
                .unwrap_or(0)
        };
        queue_kernel_event(
            state,
            KernelQueuedEvent {
                equeue: completion.equeue,
                ident: completion.ident,
                filter: KERNEL_EVENT_FILTER_AMPR,
                flags: if failure.is_some() {
                    AMPR_EVENT_FLAG_ERROR
                } else {
                    0
                },
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
        write_guest_u64(arguments[2], failure.unwrap_or(0));
    }
    if arguments[3] != 0 {
        write_guest_u32(arguments[3], id);
    }
    failure.unwrap_or(0)
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

#[cfg(test)]
mod guest_path_tests {
    use std::cell::Cell;
    use std::path::PathBuf;
    use std::sync::Arc;

    use remu_shader::GuestMemoryReader;

    use super::{ampr_read_guest_path_with, ampr_record_resolved_file};
    use crate::ProbeDispatchState;

    struct BoundedMemory {
        base: u64,
        bytes: Vec<u8>,
        maximum_read: usize,
        expose_prefix: bool,
        reads: Cell<usize>,
    }

    impl GuestMemoryReader for BoundedMemory {
        fn read(&self, address: u64, output: &mut [u8]) -> bool {
            self.reads.set(self.reads.get() + 1);
            if output.len() > self.maximum_read {
                return false;
            }
            let Some(offset) = address
                .checked_sub(self.base)
                .and_then(|offset| usize::try_from(offset).ok())
            else {
                return false;
            };
            let Some(end) = offset.checked_add(output.len()) else {
                return false;
            };
            let Some(source) = self.bytes.get(offset..end) else {
                return false;
            };
            output.copy_from_slice(source);
            true
        }

        fn readable_prefix_length(&self, address: u64, maximum_length: usize) -> Option<usize> {
            if !self.expose_prefix {
                return None;
            }
            let offset = address
                .checked_sub(self.base)
                .and_then(|offset| usize::try_from(offset).ok())?;
            Some(
                self.bytes
                    .len()
                    .saturating_sub(offset)
                    .min(maximum_length)
                    .min(self.maximum_read),
            )
        }
    }

    #[test]
    fn mapped_apr_paths_are_read_in_bounded_chunks() {
        let path = format!("/app0/{}", "asset/".repeat(100));
        let mut bytes = path.as_bytes().to_vec();
        bytes.push(0);
        let memory = BoundedMemory {
            base: 0x1000,
            bytes,
            maximum_read: 256,
            expose_prefix: true,
            reads: Cell::new(0),
        };
        assert_eq!(
            ampr_read_guest_path_with(&memory, memory.base).as_deref(),
            Some(path.as_str())
        );
        assert_eq!(memory.reads.get(), 3);
    }

    #[test]
    fn apr_path_read_falls_back_when_a_reader_cannot_describe_its_boundary() {
        let memory = BoundedMemory {
            base: 0x2000,
            bytes: b"/app0/config.ini\0".to_vec(),
            maximum_read: 1,
            expose_prefix: false,
            reads: Cell::new(0),
        };
        assert_eq!(
            ampr_read_guest_path_with(&memory, memory.base).as_deref(),
            Some("/app0/config.ini")
        );
        assert_eq!(memory.reads.get(), memory.bytes.len() + 1);
    }

    #[test]
    fn apr_path_read_rejects_unterminated_and_invalid_utf8_inputs() {
        let unterminated = BoundedMemory {
            base: 0x3000,
            bytes: vec![b'x'; super::AMPR_MAX_PATH_BYTES],
            maximum_read: 256,
            expose_prefix: true,
            reads: Cell::new(0),
        };
        assert_eq!(
            ampr_read_guest_path_with(&unterminated, unterminated.base),
            None
        );

        let invalid_utf8 = BoundedMemory {
            base: 0x4000,
            bytes: vec![0xff, 0],
            maximum_read: 256,
            expose_prefix: true,
            reads: Cell::new(0),
        };
        assert_eq!(
            ampr_read_guest_path_with(&invalid_utf8, invalid_utf8.base),
            None
        );
    }

    #[test]
    fn repeated_apr_resolution_retains_only_a_matching_open_file() {
        let file_id = 0x1234_5678;
        let original_path = PathBuf::from("/host/assets/original.pak");
        let replacement_path = PathBuf::from("/host/assets/collision.pak");
        let retained = Arc::new(std::fs::File::open(std::env::current_exe().unwrap()).unwrap());
        let mut state = ProbeDispatchState::default();
        state.ampr_files.insert(file_id, original_path.clone());
        state.ampr_file_handles.insert(file_id, retained.clone());

        ampr_record_resolved_file(&mut state, file_id, original_path);
        assert!(Arc::ptr_eq(
            state.ampr_file_handles.get(&file_id).unwrap(),
            &retained
        ));

        ampr_record_resolved_file(&mut state, file_id, replacement_path.clone());
        assert_eq!(state.ampr_files.get(&file_id), Some(&replacement_path));
        assert!(!state.ampr_file_handles.contains_key(&file_id));
    }

    #[test]
    fn distinct_paths_receive_dense_sequential_slot_ids() {
        let mut state = ProbeDispatchState::default();
        let first = state.intern_ampr_file_id(std::path::Path::new("/host/a.pak"));
        let second = state.intern_ampr_file_id(std::path::Path::new("/host/b.pak"));
        let third = state.intern_ampr_file_id(std::path::Path::new("/host/c.pak"));
        // Dense, monotonic from one: distinct files can never collide, unlike the former hash.
        assert_eq!((first, second, third), (Some(1), Some(2), Some(3)));
        // Re-resolving a known path is stable, so a title's retained identifier keeps pointing at
        // the same file across the whole run.
        assert_eq!(
            state.intern_ampr_file_id(std::path::Path::new("/host/a.pak")),
            Some(1)
        );
        assert_eq!(state.ampr_next_file_id, 3);
    }

    #[test]
    fn the_handle_cache_evicts_the_least_recently_used_beyond_the_budget() {
        let file = Arc::new(std::fs::File::open(std::env::current_exe().unwrap()).unwrap());
        let mut state = ProbeDispatchState::default();
        state.retain_ampr_file_handle_within(1, file.clone(), 2);
        state.retain_ampr_file_handle_within(2, file.clone(), 2);
        state.retain_ampr_file_handle_within(3, file.clone(), 2);
        // The oldest handle is dropped once the budget is exceeded.
        assert!(!state.ampr_file_handles.contains_key(&1));
        assert!(state.ampr_file_handles.contains_key(&2));
        assert!(state.ampr_file_handles.contains_key(&3));

        // Re-retaining an entry promotes it, so the next eviction takes the genuinely-oldest one.
        state.retain_ampr_file_handle_within(2, file.clone(), 2);
        state.retain_ampr_file_handle_within(4, file.clone(), 2);
        assert!(!state.ampr_file_handles.contains_key(&3));
        assert!(state.ampr_file_handles.contains_key(&2));
        assert!(state.ampr_file_handles.contains_key(&4));
    }
}
