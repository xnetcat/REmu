#[cfg(target_os = "macos")]
use std::sync::atomic::Ordering;
#[cfg(target_os = "macos")]
use std::sync::{Arc, RwLock};

#[cfg(target_os = "macos")]
use remu_core::GuestAddress;
#[cfg(target_os = "macos")]
use remu_memory::{AddressSpaceError, GUEST_PAGE_SIZE, GuestMapCoalescing, GuestMapPlacement};

#[cfg(target_os = "macos")]
use crate::{
    KERNEL_MEMORY_READABLE_RANGES, KERNEL_MEMORY_READABLE_RANGES_GENERATION,
    MACOS_GUEST_PAGE_ALIGNMENT, SCE_KERNEL_ERROR_MEMORY_FAULT, SCE_KERNEL_ERROR_OUT_OF_MEMORY,
    SCE_KERNEL_ERROR_PERMISSION_DENIED, SCE_KERNEL_ERROR_TRY_AGAIN, align_up_u64, guest_memset,
    kernel_memory, low_i32, read_guest_u8, read_guest_u32, read_guest_u64, trace_guest_vm,
    write_guest_u8, write_guest_u32, write_guest_u64,
};
use crate::{
    ProbeDispatchState, SCE_KERNEL_ERROR_INVALID_ARGUMENT, SCE_KERNEL_ERROR_NAME_TOO_LONG,
    SCE_KERNEL_ERROR_NOT_FOUND, read_guest_cstring, sce_kernel_error,
};

#[cfg(target_os = "macos")]
fn kernel_memory_mut(
    state: &mut ProbeDispatchState,
) -> Result<&mut kernel_memory::KernelMemory, kernel_memory::KernelMemoryError> {
    if state.kernel_memory.is_none() {
        state.kernel_memory = Some(kernel_memory::KernelMemory::new()?);
    }
    Ok(state
        .kernel_memory
        .as_mut()
        .expect("kernel memory was initialized immediately above"))
}

#[cfg(target_os = "macos")]
fn refresh_kernel_memory_readable_ranges(memory: &kernel_memory::KernelMemory) {
    *KERNEL_MEMORY_READABLE_RANGES
        .get_or_init(|| RwLock::new(Arc::from([])))
        .write()
        .unwrap_or_else(std::sync::PoisonError::into_inner) =
        Arc::from(memory.readable_ranges().into_boxed_slice());
    KERNEL_MEMORY_READABLE_RANGES_GENERATION.fetch_add(1, Ordering::Release);
}

#[cfg(target_os = "macos")]
fn kernel_memory_error(error: &kernel_memory::KernelMemoryError) -> u64 {
    use kernel_memory::KernelMemoryError;

    let code = match error {
        KernelMemoryError::DirectMemoryOverflow { .. }
        | KernelMemoryError::AddressSpace(
            AddressSpaceError::ZeroSize
            | AddressSpaceError::Unaligned { .. }
            | AddressSpaceError::InvalidAlignment { .. }
            | AddressSpaceError::AddressOverflow { .. }
            | AddressSpaceError::HostSizeOverflow { .. }
            | AddressSpaceError::BackingOutOfBounds { .. }
            | AddressSpaceError::ZeroBackingSize,
        ) => SCE_KERNEL_ERROR_INVALID_ARGUMENT,
        KernelMemoryError::DirectMemoryNotAllocated { .. }
        | KernelMemoryError::AddressSpace(
            AddressSpaceError::NotMapped { .. }
            | AddressSpaceError::Reserved { .. }
            | AddressSpaceError::Overlap { .. }
            | AddressSpaceError::OutOfBounds { .. },
        ) => SCE_KERNEL_ERROR_NOT_FOUND,
        KernelMemoryError::AddressSpace(AddressSpaceError::PermissionDenied { .. }) => {
            SCE_KERNEL_ERROR_PERMISSION_DENIED
        }
        KernelMemoryError::DirectMemoryExhausted { .. }
        | KernelMemoryError::AddressSpace(
            AddressSpaceError::InvalidHostPageSize
            | AddressSpaceError::IncompatibleHostPageSize { .. }
            | AddressSpaceError::AddressSpaceExhausted { .. }
            | AddressSpaceError::BackingOperationFailed { .. }
            | AddressSpaceError::ReservationFailed { .. }
            | AddressSpaceError::MachReservationFailed { .. }
            | AddressSpaceError::HostOperationFailed { .. },
        ) => SCE_KERNEL_ERROR_TRY_AGAIN,
    };
    sce_kernel_error(code)
}

#[cfg(target_os = "macos")]
pub(crate) fn allocate_direct_memory(arguments: [u64; 6], state: &mut ProbeDispatchState) -> u64 {
    const DIRECT_MEMORY_SIZE: u64 = kernel_memory::DIRECT_MEMORY_SIZE;
    let search_start = i64::from_ne_bytes(arguments[0].to_ne_bytes())
        .max(0)
        .cast_unsigned();
    let raw_search_end = i64::from_ne_bytes(arguments[1].to_ne_bytes());
    let length = arguments[2];
    let alignment = if arguments[3] == 0 {
        MACOS_GUEST_PAGE_ALIGNMENT
    } else {
        arguments[3]
    };
    let memory_type = low_i32(arguments[4]);
    let output = arguments[5];
    if length == 0
        || output == 0
        || !length.is_multiple_of(GUEST_PAGE_SIZE)
        || !alignment.is_power_of_two()
        || alignment < GUEST_PAGE_SIZE
    {
        return sce_kernel_error(SCE_KERNEL_ERROR_INVALID_ARGUMENT);
    }
    let search_end = if raw_search_end <= 0 {
        DIRECT_MEMORY_SIZE
    } else {
        raw_search_end.cast_unsigned().min(DIRECT_MEMORY_SIZE)
    };
    let search_start = if search_start >= search_end {
        0
    } else {
        search_start
    };
    let memory = match kernel_memory_mut(state) {
        Ok(memory) => memory,
        Err(error) => return kernel_memory_error(&error),
    };
    let offset =
        match memory.allocate_direct(search_start, search_end, length, alignment, memory_type) {
            Ok(offset) => offset,
            Err(error) => return kernel_memory_error(&error),
        };
    write_guest_u64(output, offset);
    if trace_guest_vm() {
        eprintln!(
            "[REMu][VM] allocated direct start={offset:#x} length={length:#x} alignment={alignment:#x} type={memory_type:#x}"
        );
    }
    0
}

#[cfg(not(target_os = "macos"))]
pub(crate) fn allocate_direct_memory(_arguments: [u64; 6], _state: &mut ProbeDispatchState) -> u64 {
    sce_kernel_error(SCE_KERNEL_ERROR_NOT_FOUND)
}

/// `sceKernelAvailableDirectMemorySize(searchStart, searchEnd, alignment, *start, *size)`.
///
/// Reports the largest aligned free direct-memory span in the requested window. Console memory
/// allocators call this before reserving their pool, so the arguments are normalized exactly as
/// [`allocate_direct_memory`] normalizes its own: whatever this reports, the caller's next
/// allocation of that span has to succeed.
#[cfg(target_os = "macos")]
pub(crate) fn available_direct_memory_size(
    arguments: [u64; 6],
    state: &mut ProbeDispatchState,
) -> u64 {
    const DIRECT_MEMORY_SIZE: u64 = kernel_memory::DIRECT_MEMORY_SIZE;
    let search_start = i64::from_ne_bytes(arguments[0].to_ne_bytes())
        .max(0)
        .cast_unsigned();
    let raw_search_end = i64::from_ne_bytes(arguments[1].to_ne_bytes());
    let alignment = if arguments[2] == 0 {
        MACOS_GUEST_PAGE_ALIGNMENT
    } else {
        arguments[2]
    };
    let start_output = arguments[3];
    let size_output = arguments[4];
    // Both outputs are mandatory. Returning an error while leaving them untouched is precisely
    // what makes a caller read a stale stack slot as the answer and size its pool from garbage.
    if start_output == 0
        || size_output == 0
        || !alignment.is_power_of_two()
        || alignment < GUEST_PAGE_SIZE
    {
        return sce_kernel_error(SCE_KERNEL_ERROR_INVALID_ARGUMENT);
    }
    let search_end = if raw_search_end <= 0 {
        DIRECT_MEMORY_SIZE
    } else {
        raw_search_end.cast_unsigned().min(DIRECT_MEMORY_SIZE)
    };
    let search_start = if search_start >= search_end {
        0
    } else {
        search_start
    };
    let memory = match kernel_memory_mut(state) {
        Ok(memory) => memory,
        Err(error) => return kernel_memory_error(&error),
    };
    let Some((start, size)) = memory.available_direct(search_start, search_end, alignment) else {
        return sce_kernel_error(SCE_KERNEL_ERROR_OUT_OF_MEMORY);
    };
    write_guest_u64(start_output, start);
    write_guest_u64(size_output, size);
    if trace_guest_vm() {
        eprintln!(
            "[REMu][VM] available direct start={start:#x} size={size:#x} alignment={alignment:#x}"
        );
    }
    0
}

#[cfg(not(target_os = "macos"))]
pub(crate) fn available_direct_memory_size(
    _arguments: [u64; 6],
    _state: &mut ProbeDispatchState,
) -> u64 {
    sce_kernel_error(SCE_KERNEL_ERROR_NOT_FOUND)
}

/// `sceKernelAvailableFlexibleMemorySize(*size)`.
///
/// Reports how much of the flexible-memory budget the guest has not mapped yet. Engines read this
/// once to size their own flexible pool and again while mapping into it, so it has to fall as
/// mappings are made rather than report a fixed total.
#[cfg(target_os = "macos")]
pub(crate) fn available_flexible_memory_size(
    arguments: [u64; 6],
    state: &mut ProbeDispatchState,
) -> u64 {
    let size_output = arguments[0];
    if size_output == 0 {
        return sce_kernel_error(SCE_KERNEL_ERROR_INVALID_ARGUMENT);
    }
    let memory = match kernel_memory_mut(state) {
        Ok(memory) => memory,
        Err(error) => return kernel_memory_error(&error),
    };
    let available = memory.available_flexible();
    write_guest_u64(size_output, available);
    if trace_guest_vm() {
        eprintln!("[REMu][VM] available flexible size={available:#x}");
    }
    0
}

#[cfg(not(target_os = "macos"))]
pub(crate) fn available_flexible_memory_size(
    _arguments: [u64; 6],
    _state: &mut ProbeDispatchState,
) -> u64 {
    sce_kernel_error(SCE_KERNEL_ERROR_NOT_FOUND)
}

#[cfg(target_os = "macos")]
pub(crate) fn allocate_main_direct_memory(
    arguments: [u64; 6],
    state: &mut ProbeDispatchState,
) -> u64 {
    allocate_direct_memory(
        [
            0,
            kernel_memory::DIRECT_MEMORY_SIZE,
            arguments[0],
            arguments[1],
            arguments[2],
            arguments[3],
        ],
        state,
    )
}

#[cfg(not(target_os = "macos"))]
pub(crate) fn allocate_main_direct_memory(
    _arguments: [u64; 6],
    _state: &mut ProbeDispatchState,
) -> u64 {
    sce_kernel_error(SCE_KERNEL_ERROR_NOT_FOUND)
}

#[cfg(target_os = "macos")]
pub(crate) fn direct_memory_query(arguments: [u64; 6], state: &mut ProbeDispatchState) -> u64 {
    let offset = arguments[0];
    let find_next = arguments[1] & 1 != 0;
    let output = arguments[2];
    let output_size = arguments[3];
    if output == 0 || output_size < 0x18 || offset >= kernel_memory::DIRECT_MEMORY_SIZE {
        return sce_kernel_error(SCE_KERNEL_ERROR_INVALID_ARGUMENT);
    }
    let memory = match kernel_memory_mut(state) {
        Ok(memory) => memory,
        Err(error) => return kernel_memory_error(&error),
    };
    let Some(allocation) = memory.query_direct(offset, find_next) else {
        return sce_kernel_error(SCE_KERNEL_ERROR_NOT_FOUND);
    };
    write_guest_u64(output, allocation.start);
    write_guest_u64(output + 8, allocation.end);
    write_guest_u32(output + 16, allocation.memory_type.cast_unsigned());
    0
}

#[cfg(not(target_os = "macos"))]
pub(crate) fn direct_memory_query(_arguments: [u64; 6], _state: &mut ProbeDispatchState) -> u64 {
    sce_kernel_error(SCE_KERNEL_ERROR_NOT_FOUND)
}

#[cfg(target_os = "macos")]
pub(crate) fn map_direct_memory(arguments: [u64; 6], state: &mut ProbeDispatchState) -> u64 {
    map_direct_memory_core(arguments, None, state)
}

#[cfg(target_os = "macos")]
pub(crate) fn map_direct_memory_2(
    arguments: [u64; 6],
    alignment: u64,
    state: &mut ProbeDispatchState,
) -> u64 {
    map_direct_memory_core(
        [
            arguments[0],
            arguments[1],
            arguments[3],
            arguments[4],
            arguments[5],
            alignment,
        ],
        Some(low_i32(arguments[2])),
        state,
    )
}

#[cfg(target_os = "macos")]
fn map_direct_memory_core(
    arguments: [u64; 6],
    memory_type: Option<i32>,
    state: &mut ProbeDispatchState,
) -> u64 {
    let (output, length, protection, flags, direct_start) = (
        arguments[0],
        arguments[1],
        low_i32(arguments[2]),
        arguments[3],
        arguments[4],
    );
    let alignment = if arguments[5] == 0 {
        MACOS_GUEST_PAGE_ALIGNMENT
    } else {
        arguments[5]
    };
    if output == 0
        || length == 0
        || !alignment.is_power_of_two()
        || alignment < GUEST_PAGE_SIZE
        || !direct_start.is_multiple_of(GUEST_PAGE_SIZE)
        || !length.is_multiple_of(GUEST_PAGE_SIZE)
    {
        return sce_kernel_error(SCE_KERNEL_ERROR_INVALID_ARGUMENT);
    }
    let requested = read_guest_u64(output);
    let fixed = flags & 0x10 != 0 && requested != 0;
    let memory = match kernel_memory_mut(state) {
        Ok(memory) => memory,
        Err(error) => return kernel_memory_error(&error),
    };
    let placement = if fixed {
        GuestMapPlacement::Fixed(GuestAddress::new(requested))
    } else if requested != 0 && memory.contains_virtual_address(GuestAddress::new(requested)) {
        GuestMapPlacement::Hint(GuestAddress::new(requested))
    } else {
        GuestMapPlacement::Anywhere
    };
    let coalescing = if flags & 0x40_0000 == 0 {
        GuestMapCoalescing::Allowed
    } else {
        GuestMapCoalescing::Disallowed
    };
    let result = memory.map_direct_with_options(
        placement,
        length,
        alignment,
        kernel_memory::KernelDirectMapOptions {
            raw_protection: protection,
            direct_start,
            memory_type,
            coalescing,
        },
    );
    let mapped = match result {
        Ok(mapped) => mapped,
        Err(error) => {
            if trace_guest_vm() {
                eprintln!(
                    "[REMu][VM] map direct failed direct={direct_start:#x} requested={requested:#x} length={length:#x} protection={protection:#x} flags={flags:#x} error={error}"
                );
            }
            return kernel_memory_error(&error);
        }
    };
    refresh_kernel_memory_readable_ranges(memory);
    write_guest_u64(output, mapped.get());
    if trace_guest_vm() {
        eprintln!(
            "[REMu][VM] mapped direct={direct_start:#x} address={:#x} length={length:#x} protection={protection:#x}",
            mapped.get()
        );
    }
    0
}

#[cfg(not(target_os = "macos"))]
pub(crate) fn map_direct_memory(_arguments: [u64; 6], _state: &mut ProbeDispatchState) -> u64 {
    sce_kernel_error(SCE_KERNEL_ERROR_NOT_FOUND)
}

#[cfg(not(target_os = "macos"))]
pub(crate) fn map_direct_memory_2(
    _arguments: [u64; 6],
    _alignment: u64,
    _state: &mut ProbeDispatchState,
) -> u64 {
    sce_kernel_error(SCE_KERNEL_ERROR_NOT_FOUND)
}

#[cfg(target_os = "macos")]
pub(crate) fn map_named_flexible_memory(
    arguments: [u64; 6],
    state: &mut ProbeDispatchState,
) -> u64 {
    let (output, length, protection, flags, name_address) = (
        arguments[0],
        arguments[1],
        low_i32(arguments[2]),
        arguments[3],
        arguments[4],
    );
    if output == 0 || name_address == 0 {
        return sce_kernel_error(SCE_KERNEL_ERROR_MEMORY_FAULT);
    }
    if length == 0 || !length.is_multiple_of(GUEST_PAGE_SIZE) {
        return sce_kernel_error(SCE_KERNEL_ERROR_INVALID_ARGUMENT);
    }
    let Some(name) = read_guest_cstring(name_address) else {
        return sce_kernel_error(SCE_KERNEL_ERROR_MEMORY_FAULT);
    };
    if name.len() >= 32 {
        return sce_kernel_error(SCE_KERNEL_ERROR_NAME_TOO_LONG);
    }
    map_flexible_memory_core(output, length, protection, flags, &name, state)
}

/// Shared placement and mapping behavior for every flexible-memory entry point.
///
/// The named, unnamed, and batch forms differ only in how the caller supplies the range name.
#[cfg(target_os = "macos")]
pub(crate) fn map_flexible_memory_core(
    output: u64,
    length: u64,
    protection: i32,
    flags: u64,
    name: &str,
    state: &mut ProbeDispatchState,
) -> u64 {
    let requested = read_guest_u64(output);
    let fixed = flags & 0x10 != 0;
    if fixed && requested == 0 {
        return sce_kernel_error(SCE_KERNEL_ERROR_INVALID_ARGUMENT);
    }
    let memory = match kernel_memory_mut(state) {
        Ok(memory) => memory,
        Err(error) => return kernel_memory_error(&error),
    };
    let placement = if fixed {
        GuestMapPlacement::Fixed(GuestAddress::new(requested))
    } else if requested != 0 && memory.contains_virtual_address(GuestAddress::new(requested)) {
        GuestMapPlacement::Hint(GuestAddress::new(requested))
    } else {
        GuestMapPlacement::Anywhere
    };
    let mapped =
        match memory.map_flexible(placement, length, MACOS_GUEST_PAGE_ALIGNMENT, protection) {
            Ok(mapped) => mapped,
            Err(error) => return kernel_memory_error(&error),
        };
    refresh_kernel_memory_readable_ranges(memory);
    write_guest_u64(output, mapped.get());
    if trace_guest_vm() {
        eprintln!(
            "[REMu][VM] mapped flexible address={:#x} length={length:#x} protection={protection:#x} flags={flags:#x} name={name:?}",
            mapped.get()
        );
    }
    0
}

#[cfg(not(target_os = "macos"))]
pub(crate) fn map_named_flexible_memory(
    _arguments: [u64; 6],
    _state: &mut ProbeDispatchState,
) -> u64 {
    sce_kernel_error(SCE_KERNEL_ERROR_NOT_FOUND)
}

/// Byte size of one `sceKernelBatchMap` entry: `start`, `offset`, `length`, `protection`, `type`,
/// two reserved bytes, and `operation`.
const BATCH_MAP_ENTRY_SIZE: u64 = 32;
pub(crate) const BATCH_MAP_OPERATION_MAP_DIRECT: u32 = 0;
const BATCH_MAP_OPERATION_UNMAP: u32 = 1;
const BATCH_MAP_OPERATION_PROTECT: u32 = 2;
const BATCH_MAP_OPERATION_MAP_FLEXIBLE: u32 = 3;
const BATCH_MAP_OPERATION_TYPE_PROTECT: u32 = 4;
/// `MAP_FIXED`. The single-argument batch entry point always maps at the caller's addresses.
const KERNEL_MAP_FIXED: u64 = 0x10;

#[cfg(target_os = "macos")]
pub(crate) fn kernel_batch_map(arguments: [u64; 6], state: &mut ProbeDispatchState) -> u64 {
    kernel_batch_map_core(arguments, KERNEL_MAP_FIXED, state)
}

#[cfg(target_os = "macos")]
pub(crate) fn kernel_batch_map_2(arguments: [u64; 6], state: &mut ProbeDispatchState) -> u64 {
    kernel_batch_map_core(arguments, arguments[3], state)
}

/// Applies one ordered batch of guest memory operations.
///
/// Each entry is executed in order and the batch stops at the first rejected entry, reporting how
/// many entries completed. The count is written even when the batch fails, because the guest uses
/// it to resume or unwind a partially applied batch; only a null output pointer skips it.
#[cfg(target_os = "macos")]
fn kernel_batch_map_core(arguments: [u64; 6], flags: u64, state: &mut ProbeDispatchState) -> u64 {
    let (entries, output) = (arguments[0], arguments[2]);
    let count = low_i32(arguments[1]);
    let mut result = 0;
    let mut processed = 0_i32;
    for index in 0..count.max(0) {
        let Some(entry) = u64::try_from(index)
            .ok()
            .and_then(|index| index.checked_mul(BATCH_MAP_ENTRY_SIZE))
            .and_then(|offset| entries.checked_add(offset))
            .filter(|_| entries != 0)
        else {
            result = sce_kernel_error(SCE_KERNEL_ERROR_INVALID_ARGUMENT);
            break;
        };
        let start = read_guest_u64(entry);
        let direct_offset = read_guest_u64(entry + 8);
        let length = read_guest_u64(entry + 16);
        let protection = u64::from(read_guest_u8(entry + 24));
        let memory_type = i32::from(read_guest_u8(entry + 25));
        let operation = read_guest_u32(entry + 28);
        if length == 0 || operation > BATCH_MAP_OPERATION_TYPE_PROTECT {
            result = sce_kernel_error(SCE_KERNEL_ERROR_INVALID_ARGUMENT);
            break;
        }
        result = match operation {
            // The entry's own `start` field is the in/out address, exactly as the standalone
            // mapping calls use their caller-supplied pointer.
            BATCH_MAP_OPERATION_MAP_DIRECT => map_direct_memory_core(
                [entry, length, protection, flags, direct_offset, 0],
                Some(memory_type),
                state,
            ),
            BATCH_MAP_OPERATION_UNMAP => kernel_munmap([start, length, 0, 0, 0, 0], state),
            BATCH_MAP_OPERATION_PROTECT => {
                kernel_mprotect([start, length, protection, 0, 0, 0], state)
            }
            BATCH_MAP_OPERATION_MAP_FLEXIBLE => {
                map_flexible_memory_core(entry, length, low_i32(protection), flags, "anon", state)
            }
            _ => kernel_mtypeprotect(
                [
                    start,
                    length,
                    memory_type.cast_unsigned().into(),
                    protection,
                    0,
                    0,
                ],
                state,
            ),
        };
        if result != 0 {
            break;
        }
        processed += 1;
    }
    if output != 0 {
        write_guest_u32(output, processed.cast_unsigned());
    }
    result
}

#[cfg(not(target_os = "macos"))]
pub(crate) fn kernel_batch_map(_arguments: [u64; 6], _state: &mut ProbeDispatchState) -> u64 {
    sce_kernel_error(SCE_KERNEL_ERROR_NOT_FOUND)
}

#[cfg(not(target_os = "macos"))]
pub(crate) fn kernel_batch_map_2(_arguments: [u64; 6], _state: &mut ProbeDispatchState) -> u64 {
    sce_kernel_error(SCE_KERNEL_ERROR_NOT_FOUND)
}

/// Reprotects a range and assigns it a new guest memory type in the same operation.
#[cfg(target_os = "macos")]
pub(crate) fn kernel_mtypeprotect(arguments: [u64; 6], state: &mut ProbeDispatchState) -> u64 {
    let (address, length) = (arguments[0], arguments[1]);
    let memory_type = low_i32(arguments[2]);
    let protection = low_i32(arguments[3]);
    if address == 0 || length == 0 {
        return sce_kernel_error(SCE_KERNEL_ERROR_INVALID_ARGUMENT);
    }
    let aligned_address = address & !(GUEST_PAGE_SIZE - 1);
    let Some(aligned_end) = address
        .checked_add(length)
        .and_then(|end| align_up_u64(end, GUEST_PAGE_SIZE))
    else {
        return sce_kernel_error(SCE_KERNEL_ERROR_INVALID_ARGUMENT);
    };
    let memory = match kernel_memory_mut(state) {
        Ok(memory) => memory,
        Err(error) => return kernel_memory_error(&error),
    };
    let committed = match memory.protect_with_type(
        GuestAddress::new(aligned_address),
        aligned_end - aligned_address,
        protection,
        Some(memory_type),
    ) {
        Ok(committed) => committed,
        Err(error) => return kernel_memory_error(&error),
    };
    refresh_kernel_memory_readable_ranges(memory);
    if trace_guest_vm() {
        eprintln!(
            "[REMu][VM] type-protected address={aligned_address:#x} length={:#x} type={memory_type:#x} protection={protection:#x} committed={committed:#x}",
            aligned_end - aligned_address
        );
    }
    0
}

#[cfg(not(target_os = "macos"))]
pub(crate) fn kernel_mtypeprotect(_arguments: [u64; 6], _state: &mut ProbeDispatchState) -> u64 {
    sce_kernel_error(SCE_KERNEL_ERROR_NOT_FOUND)
}

#[cfg(target_os = "macos")]
pub(crate) fn kernel_reserve_virtual_range(
    arguments: [u64; 6],
    state: &mut ProbeDispatchState,
) -> u64 {
    let output = arguments[0];
    let length = arguments[1];
    let flags = low_i32(arguments[2]);
    let alignment = if arguments[3] == 0 {
        MACOS_GUEST_PAGE_ALIGNMENT
    } else {
        arguments[3]
    };
    if output == 0
        || length == 0
        || !length.is_multiple_of(GUEST_PAGE_SIZE)
        || !alignment.is_power_of_two()
        || alignment < GUEST_PAGE_SIZE
    {
        return sce_kernel_error(SCE_KERNEL_ERROR_INVALID_ARGUMENT);
    }

    let requested = read_guest_u64(output);
    let memory = match kernel_memory_mut(state) {
        Ok(memory) => memory,
        Err(error) => return kernel_memory_error(&error),
    };
    let placement = if flags & 0x10 != 0 && requested != 0 {
        GuestMapPlacement::Fixed(GuestAddress::new(requested))
    } else if requested != 0 && memory.contains_virtual_address(GuestAddress::new(requested)) {
        GuestMapPlacement::Hint(GuestAddress::new(requested))
    } else {
        GuestMapPlacement::Anywhere
    };
    let reserved = match memory.reserve_virtual(placement, length, alignment) {
        Ok(address) => address,
        Err(error) => return kernel_memory_error(&error),
    };
    write_guest_u64(output, reserved.get());
    if trace_guest_vm() {
        eprintln!(
            "[REMu][VM] reserved address={:#x} length={length:#x} alignment={alignment:#x} flags={flags:#x}",
            reserved.get()
        );
    }
    0
}

#[cfg(not(target_os = "macos"))]
pub(crate) fn kernel_reserve_virtual_range(
    _arguments: [u64; 6],
    _state: &mut ProbeDispatchState,
) -> u64 {
    sce_kernel_error(SCE_KERNEL_ERROR_NOT_FOUND)
}

pub(crate) fn kernel_set_virtual_range_name(arguments: [u64; 6]) -> u64 {
    if arguments[0] == 0 || arguments[1] == 0 || arguments[2] == 0 {
        return sce_kernel_error(SCE_KERNEL_ERROR_INVALID_ARGUMENT);
    }
    match read_guest_cstring(arguments[2]) {
        Some(name) if !name.is_empty() => 0,
        _ => sce_kernel_error(SCE_KERNEL_ERROR_INVALID_ARGUMENT),
    }
}

#[cfg(target_os = "macos")]
pub(crate) fn kernel_mprotect(arguments: [u64; 6], state: &mut ProbeDispatchState) -> u64 {
    let address = arguments[0];
    let length = arguments[1];
    let protection = low_i32(arguments[2]);
    if address == 0 || length == 0 {
        return sce_kernel_error(SCE_KERNEL_ERROR_INVALID_ARGUMENT);
    }
    let aligned_address = address & !(GUEST_PAGE_SIZE - 1);
    let Some(raw_end) = address.checked_add(length) else {
        return sce_kernel_error(SCE_KERNEL_ERROR_INVALID_ARGUMENT);
    };
    let Some(aligned_end) = align_up_u64(raw_end, GUEST_PAGE_SIZE) else {
        return sce_kernel_error(SCE_KERNEL_ERROR_INVALID_ARGUMENT);
    };
    let aligned_length = aligned_end - aligned_address;
    if process_mapping_contains(
        state.process_mapping_start,
        state.process_mapping_end,
        aligned_address,
        aligned_end,
    ) {
        // The Metal backend may have armed CPU-write tracking for a unified-memory texture in
        // this process reservation. Distinguish a permission-only change from replacement so the
        // backend can preserve GPU-private contents and install a fresh watch afterwards.
        let reprotection = remu_memory::CpuWriteWatch::begin_reprotect_overlapping(
            aligned_address,
            aligned_length,
        );
        let Ok(host_address) = usize::try_from(aligned_address) else {
            return sce_kernel_error(SCE_KERNEL_ERROR_INVALID_ARGUMENT);
        };
        let Ok(host_length) = usize::try_from(aligned_length) else {
            return sce_kernel_error(SCE_KERNEL_ERROR_INVALID_ARGUMENT);
        };
        // SAFETY: The range was page-aligned and bounds-checked against the live process image.
        let result = unsafe {
            libc::mprotect(
                host_address as *mut libc::c_void,
                host_length,
                host_protection_from_guest(protection),
            )
        };
        drop(reprotection);
        if result != 0 {
            return sce_kernel_error(SCE_KERNEL_ERROR_NOT_FOUND);
        }
        if trace_guest_vm() {
            eprintln!(
                "[REMu][VM] protected process image address={aligned_address:#x} length={aligned_length:#x} protection={protection:#x}"
            );
        }
        return 0;
    }
    let memory = match kernel_memory_mut(state) {
        Ok(memory) => memory,
        Err(error) => return kernel_memory_error(&error),
    };
    let committed = match memory.protect(
        GuestAddress::new(aligned_address),
        aligned_length,
        protection,
    ) {
        Ok(committed) => committed,
        Err(error) => return kernel_memory_error(&error),
    };
    refresh_kernel_memory_readable_ranges(memory);
    if trace_guest_vm() {
        eprintln!(
            "[REMu][VM] protected address={aligned_address:#x} length={aligned_length:#x} protection={protection:#x} committed={committed:#x}"
        );
    }
    0
}

#[cfg(target_os = "macos")]
pub(crate) const fn process_mapping_contains(
    mapping_start: u64,
    mapping_end: u64,
    range_start: u64,
    range_end: u64,
) -> bool {
    mapping_start != 0
        && mapping_end > mapping_start
        && range_end > range_start
        && range_start >= mapping_start
        && range_end <= mapping_end
}

#[cfg(target_os = "macos")]
pub(crate) const fn host_protection_from_guest(raw: i32) -> i32 {
    let mut protection = libc::PROT_NONE;
    if raw & (0x01 | 0x02 | 0x10) != 0 {
        protection |= libc::PROT_READ;
    }
    if raw & (0x02 | 0x20) != 0 {
        protection |= libc::PROT_WRITE;
    }
    if raw & 0x04 != 0 {
        protection |= libc::PROT_EXEC;
    }
    protection
}

#[cfg(not(target_os = "macos"))]
pub(crate) fn kernel_mprotect(_arguments: [u64; 6], _state: &mut ProbeDispatchState) -> u64 {
    sce_kernel_error(SCE_KERNEL_ERROR_NOT_FOUND)
}

#[cfg(target_os = "macos")]
pub(crate) fn kernel_virtual_query(arguments: [u64; 6], state: &ProbeDispatchState) -> u64 {
    const VIRTUAL_QUERY_INFO_SIZE: u64 = 72;
    let (query, flags, output, output_size) = (
        arguments[0],
        low_i32(arguments[1]),
        arguments[2],
        arguments[3],
    );
    if output == 0 || output_size != VIRTUAL_QUERY_INFO_SIZE || !matches!(flags, 0 | 1) {
        return sce_kernel_error(SCE_KERNEL_ERROR_INVALID_ARGUMENT);
    }
    let Some(memory) = state.kernel_memory.as_ref() else {
        return sce_kernel_error(SCE_KERNEL_ERROR_NOT_FOUND);
    };
    let Some(allocation) = memory.query(GuestAddress::new(query), flags & 1 != 0) else {
        return sce_kernel_error(SCE_KERNEL_ERROR_NOT_FOUND);
    };
    let end = allocation.address.get().saturating_add(allocation.size);
    guest_memset(output, 0, VIRTUAL_QUERY_INFO_SIZE);
    write_guest_u64(output, allocation.address.get());
    write_guest_u64(output + 0x08, end);
    write_guest_u64(output + 0x10, allocation.direct_start);
    write_guest_u32(output + 0x18, allocation.protection.cast_unsigned());
    write_guest_u32(output + 0x1c, allocation.memory_type.cast_unsigned());
    let (mapping_flags, name): (u8, &[u8]) = match allocation.kind {
        kernel_memory::KernelVirtualMappingKind::Reserved => (0x10, b"reserved"),
        kernel_memory::KernelVirtualMappingKind::Flexible => (0x11, b"flexible"),
        kernel_memory::KernelVirtualMappingKind::Direct => (0x12, b"direct"),
    };
    write_guest_u8(output + 0x20, mapping_flags);
    for (index, byte) in name.iter().copied().enumerate() {
        // `VirtualQueryInfo` stores all mapping booleans in the complete dword at 0x20. The
        // 32-byte name begins at 0x24; starting it at 0x21 corrupts the remaining bitfield bytes
        // and shifts the name three bytes ahead of the ABI consumed by Unreal.
        write_guest_u8(output + 0x24 + u64::try_from(index).unwrap_or(0), byte);
    }
    if trace_guest_vm() {
        eprintln!(
            "[REMu][VM] query={query:#x} start={:#x} end={end:#x} direct={:#x}",
            allocation.address.get(),
            allocation.direct_start
        );
    }
    0
}

#[cfg(not(target_os = "macos"))]
pub(crate) fn kernel_virtual_query(_arguments: [u64; 6], _state: &ProbeDispatchState) -> u64 {
    sce_kernel_error(SCE_KERNEL_ERROR_NOT_FOUND)
}

#[cfg(target_os = "macos")]
pub(crate) fn kernel_munmap(arguments: [u64; 6], state: &mut ProbeDispatchState) -> u64 {
    let (address, length) = (arguments[0], arguments[1]);
    if address == 0
        || length == 0
        || !address.is_multiple_of(GUEST_PAGE_SIZE)
        || !length.is_multiple_of(GUEST_PAGE_SIZE)
    {
        return sce_kernel_error(SCE_KERNEL_ERROR_INVALID_ARGUMENT);
    }
    if address.checked_add(length).is_none() {
        return sce_kernel_error(SCE_KERNEL_ERROR_INVALID_ARGUMENT);
    }
    let Some(memory) = state.kernel_memory.as_mut() else {
        return sce_kernel_error(SCE_KERNEL_ERROR_NOT_FOUND);
    };
    if let Err(error) = memory.unmap(GuestAddress::new(address), length) {
        return kernel_memory_error(&error);
    }
    refresh_kernel_memory_readable_ranges(memory);
    if trace_guest_vm() {
        eprintln!("[REMu][VM] unmapped address={address:#x} length={length:#x}");
    }
    0
}

#[cfg(not(target_os = "macos"))]
pub(crate) fn kernel_munmap(_arguments: [u64; 6], _state: &mut ProbeDispatchState) -> u64 {
    sce_kernel_error(SCE_KERNEL_ERROR_NOT_FOUND)
}

#[cfg(target_os = "macos")]
pub(crate) fn kernel_release_direct_memory(
    arguments: [u64; 6],
    state: &mut ProbeDispatchState,
) -> u64 {
    kernel_release_direct_memory_core(arguments, false, state)
}

#[cfg(target_os = "macos")]
pub(crate) fn kernel_checked_release_direct_memory(
    arguments: [u64; 6],
    state: &mut ProbeDispatchState,
) -> u64 {
    kernel_release_direct_memory_core(arguments, true, state)
}

#[cfg(target_os = "macos")]
fn kernel_release_direct_memory_core(
    arguments: [u64; 6],
    checked: bool,
    state: &mut ProbeDispatchState,
) -> u64 {
    let (start, length) = (arguments[0], arguments[1]);
    if !start.is_multiple_of(GUEST_PAGE_SIZE) || !length.is_multiple_of(GUEST_PAGE_SIZE) {
        return sce_kernel_error(SCE_KERNEL_ERROR_INVALID_ARGUMENT);
    }
    if length == 0 {
        return 0;
    }
    if start.checked_add(length).is_none() {
        return sce_kernel_error(SCE_KERNEL_ERROR_INVALID_ARGUMENT);
    }
    let released = if let Some(memory) = state.kernel_memory.as_mut() {
        match memory.release_direct(start, length) {
            Ok(released) => {
                refresh_kernel_memory_readable_ranges(memory);
                released
            }
            Err(error) => return kernel_memory_error(&error),
        }
    } else {
        false
    };
    if checked && !released {
        return sce_kernel_error(SCE_KERNEL_ERROR_NOT_FOUND);
    }
    if trace_guest_vm() {
        eprintln!("[REMu][VM] released direct start={start:#x} length={length:#x}");
    }
    0
}

#[cfg(not(target_os = "macos"))]
pub(crate) fn kernel_release_direct_memory(
    _arguments: [u64; 6],
    _state: &mut ProbeDispatchState,
) -> u64 {
    sce_kernel_error(SCE_KERNEL_ERROR_NOT_FOUND)
}

#[cfg(not(target_os = "macos"))]
pub(crate) fn kernel_checked_release_direct_memory(
    _arguments: [u64; 6],
    _state: &mut ProbeDispatchState,
) -> u64 {
    sce_kernel_error(SCE_KERNEL_ERROR_NOT_FOUND)
}
