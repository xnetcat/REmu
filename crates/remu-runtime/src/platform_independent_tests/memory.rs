// Copyright (C) 2026 REmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#[cfg(any(target_os = "macos", windows))]
use super::batch_map_entry;
use super::{
    BATCH_MAP_OPERATION_MAP_DIRECT, ProbeDispatchState, SCE_KERNEL_ERROR_INVALID_ARGUMENT,
    SCE_KERNEL_ERROR_NOT_FOUND, SCE_KERNEL_ERROR_OUT_OF_MEMORY, allocate_direct_memory,
    available_direct_memory_size, available_flexible_memory_size, kernel_batch_map,
    kernel_release_direct_memory, kernel_virtual_query, sce_kernel_error,
};

#[cfg(any(target_os = "macos", windows))]
#[test]
fn batch_map_applies_entries_in_order_and_reports_the_completed_count() {
    let mut state = ProbeDispatchState::default();
    let mut physical = 0_u64;
    // A non-positive search end means the whole direct-memory range.
    assert_eq!(
        allocate_direct_memory(
            [0, 0, 0x8000, 0x4000, 12, (&raw mut physical).addr() as u64,],
            &mut state,
        ),
        0
    );
    let mut entries = [
        batch_map_entry(0, 0, 0x4000, 0x03, 12, BATCH_MAP_OPERATION_MAP_DIRECT),
        batch_map_entry(0, 0x4000, 0x4000, 0x03, 12, BATCH_MAP_OPERATION_MAP_DIRECT),
    ];
    let mut completed = 0_u32;
    assert_eq!(
        kernel_batch_map(
            [
                entries.as_mut_ptr().addr() as u64,
                2,
                (&raw mut completed).addr() as u64,
                0,
                0,
                0,
            ],
            &mut state,
        ),
        0
    );
    assert_eq!(completed, 2);
    // Each entry's own start field receives the address the kernel chose for it.
    let first = u64::from_le_bytes(entries[0][0..8].try_into().unwrap());
    let second = u64::from_le_bytes(entries[1][0..8].try_into().unwrap());
    assert_ne!(first, 0);
    assert_ne!(second, 0);
    assert_ne!(first, second);
}

#[cfg(any(target_os = "macos", windows))]
#[test]
fn batch_map_stops_at_the_first_invalid_entry_and_still_reports_progress() {
    let mut state = ProbeDispatchState::default();
    let mut physical = 0_u64;
    assert_eq!(
        allocate_direct_memory(
            [0, 0, 0x8000, 0x4000, 12, (&raw mut physical).addr() as u64,],
            &mut state,
        ),
        0
    );
    let mut entries = [
        batch_map_entry(0, 0, 0x4000, 0x03, 12, BATCH_MAP_OPERATION_MAP_DIRECT),
        // An operation outside the defined set is rejected before anything is applied.
        batch_map_entry(0, 0x4000, 0x4000, 0x03, 12, 5),
        batch_map_entry(0, 0x4000, 0x4000, 0x03, 12, BATCH_MAP_OPERATION_MAP_DIRECT),
    ];
    let mut completed = 0_u32;
    assert_eq!(
        kernel_batch_map(
            [
                entries.as_mut_ptr().addr() as u64,
                3,
                (&raw mut completed).addr() as u64,
                0,
                0,
                0,
            ],
            &mut state,
        ),
        sce_kernel_error(SCE_KERNEL_ERROR_INVALID_ARGUMENT)
    );
    assert_eq!(completed, 1);
    // The entry after the rejected one is never applied.
    assert_eq!(u64::from_le_bytes(entries[2][0..8].try_into().unwrap()), 0);
}

#[cfg(any(target_os = "macos", windows))]
#[test]
fn maps_named_flexible_memory_with_cpu_read_write_protection() {
    let mut state = ProbeDispatchState::default();
    let mut mapped = 0_u64;
    let name = b"CppWebApi\0";
    assert_eq!(
        crate::map_named_flexible_memory(
            [
                (&raw mut mapped).addr() as u64,
                0x8_0000,
                0x02,
                0,
                name.as_ptr().addr() as u64,
                0,
            ],
            &mut state,
        ),
        0
    );
    assert_ne!(mapped, 0);
    // SAFETY: protection 2 is the guest ABI's CPU read/write mode.
    unsafe {
        (mapped as *mut u8).write_volatile(0x5a);
        assert_eq!((mapped as *const u8).read_volatile(), 0x5a);
    }

    let mut output = [0xa5_u8; 72];
    assert_eq!(
        kernel_virtual_query(
            [
                mapped,
                0,
                output.as_mut_ptr().addr() as u64,
                output.len() as u64,
                0,
                0,
            ],
            &state,
        ),
        0
    );
    assert_eq!(u32::from_le_bytes(output[24..28].try_into().unwrap()), 2);
    assert_eq!(u32::from_le_bytes(output[32..36].try_into().unwrap()), 0x11);
    assert_eq!(&output[36..44], b"flexible");
    assert_eq!(
        crate::known_nid_name("mL8NDH86iQI"),
        Some("sceKernelMapNamedFlexibleMemory")
    );
}

#[cfg(any(target_os = "macos", windows))]
#[test]
fn rejects_invalid_named_flexible_memory_arguments() {
    let mut state = ProbeDispatchState::default();
    let mut mapped = 0_u64;
    let short_name = b"flex\0";
    let long_name = [b'x'; 33];
    let mut terminated_long_name = [0_u8; 34];
    terminated_long_name[..33].copy_from_slice(&long_name);

    assert_eq!(
        crate::map_named_flexible_memory(
            [
                (&raw mut mapped).addr() as u64,
                1,
                2,
                0,
                short_name.as_ptr().addr() as u64,
                0,
            ],
            &mut state,
        ),
        sce_kernel_error(crate::SCE_KERNEL_ERROR_INVALID_ARGUMENT)
    );
    assert_eq!(
        crate::map_named_flexible_memory(
            [
                (&raw mut mapped).addr() as u64,
                0x4000,
                2,
                0,
                terminated_long_name.as_ptr().addr() as u64,
                0,
            ],
            &mut state,
        ),
        sce_kernel_error(crate::SCE_KERNEL_ERROR_NAME_TOO_LONG)
    );
    assert_eq!(
        crate::map_named_flexible_memory(
            [
                (&raw mut mapped).addr() as u64,
                0x4000,
                2,
                0x10,
                short_name.as_ptr().addr() as u64,
                0,
            ],
            &mut state,
        ),
        sce_kernel_error(crate::SCE_KERNEL_ERROR_INVALID_ARGUMENT)
    );
}

#[cfg(any(target_os = "macos", windows))]
#[test]
fn mspace_allocates_inside_named_flexible_memory_and_reuses_freed_blocks() {
    let mut state = ProbeDispatchState::default();
    let mut mapped = 0_u64;
    let mapping_name = b"WebApiBacking\0";
    assert_eq!(
        crate::map_named_flexible_memory(
            [
                (&raw mut mapped).addr() as u64,
                0x8000,
                2,
                0,
                mapping_name.as_ptr().addr() as u64,
                0,
            ],
            &mut state,
        ),
        0
    );
    let mspace_name = b"WebApiMspace\0";
    let handle = crate::guest_mspace_create(
        [mspace_name.as_ptr().addr() as u64, mapped, 0x8000, 0, 0, 0],
        &mut state,
    );
    assert_eq!(handle, mapped);

    let first = crate::guest_mspace_malloc([handle, 0x58, 0, 0, 0, 0], &mut state);
    let second = crate::guest_mspace_malloc([handle, 0x30, 0, 0, 0, 0], &mut state);
    assert!(first >= mapped + 0x100);
    assert_eq!(second, first + 0x60);
    // SAFETY: `first` is a live mspace allocation of 0x60 usable bytes.
    unsafe { std::ptr::write_bytes(first as *mut u8, 0x5a, 0x58) };
    let grown = crate::guest_mspace_realloc([handle, first, 0x100, 0, 0, 0], &mut state);
    assert_ne!(grown, 0);
    // SAFETY: realloc preserved the first 0x58 bytes in the returned live allocation.
    assert!(
        unsafe { std::slice::from_raw_parts(grown as *const u8, 0x58) }
            .iter()
            .all(|byte| *byte == 0x5a)
    );
    assert_eq!(
        crate::guest_mspace_usable_size([handle, grown, 0, 0, 0, 0], &state),
        0x100
    );

    crate::guest_mspace_free([handle, second, 0, 0, 0, 0], &mut state);
    crate::guest_mspace_free([handle, grown, 0, 0, 0, 0], &mut state);
    assert_eq!(
        crate::guest_mspace_is_heap_empty([handle, 0, 0, 0, 0, 0], &state),
        1
    );
    assert_eq!(
        crate::guest_mspace_malloc([handle, 0x58, 0, 0, 0, 0], &mut state),
        first
    );
    assert_eq!(
        crate::known_nid_name("OJjm-QOIHlI"),
        Some("sceLibcMspaceMalloc")
    );
}

#[cfg(any(target_os = "macos", windows))]
#[test]
fn available_direct_memory_size_reports_a_span_the_allocator_grants() {
    let mut state = ProbeDispatchState::default();
    // The failing title reserves its allocator tracking overhead at the top of the pool, asks how
    // much direct memory is left, and pre-allocates exactly that. Reporting nothing here left it
    // computing a negative pre-allocation size and aborting in its own page allocator.
    const OVERHEAD: u64 = 0x26_0000;
    const ALIGNMENT: u64 = 0x1_0000;
    const POOL: u64 = 0x4_0000_0000;
    let mut tracking = 0_u64;
    assert_eq!(
        allocate_direct_memory(
            [
                POOL - OVERHEAD,
                POOL,
                OVERHEAD,
                ALIGNMENT,
                12,
                (&raw mut tracking).addr() as u64,
            ],
            &mut state,
        ),
        0
    );
    assert_eq!(tracking, POOL - OVERHEAD);
    let mut physical = u64::MAX;
    let mut size = u64::MAX;
    assert_eq!(
        available_direct_memory_size(
            [
                0,
                POOL,
                ALIGNMENT,
                (&raw mut physical).addr() as u64,
                (&raw mut size).addr() as u64,
                0,
            ],
            &mut state,
        ),
        0
    );
    assert_eq!(physical, 0);
    assert_eq!(size, POOL - OVERHEAD);
    let mut preallocated = 0_u64;
    assert_eq!(
        allocate_direct_memory(
            [
                physical,
                physical + size,
                size,
                ALIGNMENT,
                12,
                (&raw mut preallocated).addr() as u64,
            ],
            &mut state,
        ),
        0
    );
    assert_eq!(preallocated, 0);
}

#[cfg(any(target_os = "macos", windows))]
#[test]
fn available_direct_memory_size_rejects_missing_outputs_and_an_exhausted_window() {
    let mut state = ProbeDispatchState::default();
    let mut value = 0_u64;
    // Both outputs are mandatory: a caller that cannot receive the span must not read a stale
    // stack slot as if it were an answer.
    assert_eq!(
        available_direct_memory_size(
            [
                0,
                0x4_0000_0000,
                0x1_0000,
                0,
                (&raw mut value).addr() as u64,
                0
            ],
            &mut state,
        ),
        sce_kernel_error(SCE_KERNEL_ERROR_INVALID_ARGUMENT)
    );
    assert_eq!(
        available_direct_memory_size(
            [
                0,
                0x4_0000_0000,
                0x1_0000,
                (&raw mut value).addr() as u64,
                0,
                0
            ],
            &mut state,
        ),
        sce_kernel_error(SCE_KERNEL_ERROR_INVALID_ARGUMENT)
    );
    // An alignment that is not a page-aligned power of two cannot describe a real span.
    assert_eq!(
        available_direct_memory_size(
            [
                0,
                0x4_0000_0000,
                0x3000,
                (&raw mut value).addr() as u64,
                (&raw mut value).addr() as u64,
                0,
            ],
            &mut state,
        ),
        sce_kernel_error(SCE_KERNEL_ERROR_INVALID_ARGUMENT)
    );
    let mut physical = 0_u64;
    assert_eq!(
        allocate_direct_memory(
            [
                0,
                0x4_0000,
                0x4_0000,
                0x4000,
                12,
                (&raw mut physical).addr() as u64
            ],
            &mut state,
        ),
        0
    );
    // A window with no free bytes left reports exhaustion rather than a zero-length span.
    let mut size = u64::MAX;
    assert_eq!(
        available_direct_memory_size(
            [
                0,
                0x4_0000,
                0x4000,
                (&raw mut physical).addr() as u64,
                (&raw mut size).addr() as u64,
                0,
            ],
            &mut state,
        ),
        sce_kernel_error(SCE_KERNEL_ERROR_OUT_OF_MEMORY)
    );
    assert_eq!(size, u64::MAX);
}

#[cfg(any(target_os = "macos", windows))]
#[test]
fn available_flexible_memory_size_shrinks_as_flexible_memory_is_mapped() {
    let mut state = ProbeDispatchState::default();
    let mut available = 0_u64;
    assert_eq!(
        available_flexible_memory_size(
            [(&raw mut available).addr() as u64, 0, 0, 0, 0, 0],
            &mut state
        ),
        0
    );
    let unused = available;
    assert_ne!(unused, 0);
    let mut mapped = 0_u64;
    let name = b"flexpool\0";
    assert_eq!(
        crate::map_named_flexible_memory(
            [
                (&raw mut mapped).addr() as u64,
                0x8_0000,
                0x02,
                0,
                name.as_ptr().addr() as u64,
                0,
            ],
            &mut state,
        ),
        0
    );
    // The budget has to reflect what the guest actually took, or a caller that maps until the
    // pool is empty never terminates.
    assert_eq!(
        available_flexible_memory_size(
            [(&raw mut available).addr() as u64, 0, 0, 0, 0, 0],
            &mut state
        ),
        0
    );
    assert_eq!(available, unused - 0x8_0000);
    // The output pointer is mandatory.
    assert_eq!(
        available_flexible_memory_size([0, 0, 0, 0, 0, 0], &mut state),
        sce_kernel_error(SCE_KERNEL_ERROR_INVALID_ARGUMENT)
    );
    assert_eq!(
        crate::known_nid_name("aNz11fnnzi4"),
        Some("sceKernelAvailableFlexibleMemorySize")
    );
}

#[cfg(any(target_os = "macos", windows))]
#[test]
fn describes_a_tracked_direct_mapping_to_the_guest() {
    let mut state = ProbeDispatchState::default();
    let mut direct = u64::MAX;
    assert_eq!(
        allocate_direct_memory(
            [
                0x20_0000,
                0x21_0000,
                0x1_0000,
                0x4000,
                12,
                (&raw mut direct).addr() as u64,
            ],
            &mut state,
        ),
        0
    );
    assert_eq!(direct, 0x20_0000);
    let mut mapped = 0_u64;
    assert_eq!(
        crate::map_direct_memory(
            [
                (&raw mut mapped).addr() as u64,
                0x1_0000,
                0x33,
                0,
                direct,
                0x4000,
            ],
            &mut state,
        ),
        0
    );
    let mut output = [0xa5_u8; 72];
    let result = kernel_virtual_query(
        [
            mapped,
            0,
            output.as_mut_ptr().addr() as u64,
            output.len() as u64,
            0,
            0,
        ],
        &state,
    );
    assert_eq!(result, 0);
    assert_eq!(u64::from_le_bytes(output[0..8].try_into().unwrap()), mapped);
    assert_eq!(
        u64::from_le_bytes(output[8..16].try_into().unwrap()),
        mapped + 0x1_0000
    );
    assert_eq!(
        u64::from_le_bytes(output[16..24].try_into().unwrap()),
        0x20_0000
    );
    assert_eq!(u32::from_le_bytes(output[24..28].try_into().unwrap()), 0x33);
    assert_eq!(u32::from_le_bytes(output[28..32].try_into().unwrap()), 12);
    assert_eq!(u32::from_le_bytes(output[32..36].try_into().unwrap()), 0x12);
    assert_eq!(&output[36..42], b"direct");
    assert_eq!(&output[42..], &[0; 30]);
}

#[cfg(any(target_os = "macos", windows))]
#[test]
fn allocates_direct_memory_from_real_free_ranges() {
    let mut state = ProbeDispatchState::default();
    let mut occupied = u64::MAX;
    assert_eq!(
        allocate_direct_memory(
            [
                0x4000,
                0x8000,
                0x4000,
                0x4000,
                3,
                (&raw mut occupied).addr() as u64,
            ],
            &mut state,
        ),
        0
    );
    assert_eq!(occupied, 0x4000);
    let mut output = u64::MAX;
    let result = allocate_direct_memory(
        [
            0,
            0x1_0000,
            0x4000,
            0x4000,
            12,
            (&raw mut output).addr() as u64,
        ],
        &mut state,
    );
    assert_eq!(result, 0);
    assert_eq!(output, 0);
}

#[cfg(any(target_os = "macos", windows))]
#[test]
fn release_direct_memory_splits_the_owned_physical_range() {
    let mut state = ProbeDispatchState::default();
    let mut allocated = u64::MAX;
    assert_eq!(
        allocate_direct_memory(
            [
                0,
                0xc000,
                0xc000,
                0x4000,
                7,
                (&raw mut allocated).addr() as u64,
            ],
            &mut state,
        ),
        0
    );
    assert_eq!(allocated, 0);
    assert_eq!(
        kernel_release_direct_memory([0x4000, 0x4000, 0, 0, 0, 0], &mut state),
        0
    );
    let mut replacement = u64::MAX;
    assert_eq!(
        allocate_direct_memory(
            [
                0x4000,
                0x8000,
                0x4000,
                0x4000,
                9,
                (&raw mut replacement).addr() as u64,
            ],
            &mut state,
        ),
        0
    );
    assert_eq!(replacement, 0x4000);
}

#[cfg(any(target_os = "macos", windows))]
#[test]
fn direct_memory_mappings_alias_the_same_physical_bytes() {
    use crate::{kernel_munmap, map_direct_memory};

    let mut state = ProbeDispatchState::default();
    let mut direct = u64::MAX;
    assert_eq!(
        allocate_direct_memory(
            [
                0,
                0x4000,
                0x4000,
                0x4000,
                12,
                (&raw mut direct).addr() as u64,
            ],
            &mut state,
        ),
        0
    );
    assert_eq!(direct, 0);
    let mut first = 0_u64;
    let mut second = 0_u64;
    assert_eq!(
        map_direct_memory(
            [(&raw mut first).addr() as u64, 0x4000, 0x33, 0, 0, 0x4000,],
            &mut state,
        ),
        0
    );
    assert_eq!(
        map_direct_memory(
            [(&raw mut second).addr() as u64, 0x4000, 0x33, 0, 0, 0x4000,],
            &mut state,
        ),
        0
    );
    // SAFETY: both pointers are live read/write mappings created immediately above.
    unsafe {
        (first as *mut u8).write_volatile(0x5a);
        assert_eq!((second as *const u8).read_volatile(), 0x5a);
    }
    assert_eq!(kernel_munmap([first, 0x4000, 0, 0, 0, 0], &mut state), 0);
    assert_eq!(kernel_munmap([second, 0x4000, 0, 0, 0, 0], &mut state), 0);
}

#[cfg(any(target_os = "macos", windows))]
#[test]
fn direct_memory_abi_honors_fixed_and_hinted_placement_without_overwrite() {
    let mut state = ProbeDispatchState::default();
    let mut direct = u64::MAX;
    assert_eq!(
        allocate_direct_memory(
            [
                0,
                0xc000,
                0xc000,
                0x4000,
                5,
                (&raw mut direct).addr() as u64,
            ],
            &mut state,
        ),
        0
    );
    let base = state.kernel_memory.as_ref().unwrap().virtual_base();
    let mut fixed = base.get() + 4 * 0x4000;
    assert_eq!(
        crate::map_direct_memory(
            [
                (&raw mut fixed).addr() as u64,
                0x4000,
                0x33,
                0x10,
                direct,
                0x4000,
            ],
            &mut state,
        ),
        0
    );
    let mut collision = fixed;
    assert_eq!(
        crate::map_direct_memory(
            [
                (&raw mut collision).addr() as u64,
                0x4000,
                0x33,
                0x10,
                direct + 0x4000,
                0x4000,
            ],
            &mut state,
        ),
        sce_kernel_error(SCE_KERNEL_ERROR_NOT_FOUND)
    );
    let mut hinted = base.get() + 8 * 0x4000;
    let requested_hint = hinted;
    assert_eq!(
        crate::map_direct_memory(
            [
                (&raw mut hinted).addr() as u64,
                0x4000,
                0x33,
                0,
                direct + 0x8000,
                0x4000,
            ],
            &mut state,
        ),
        0
    );
    assert_eq!(hinted, requested_hint);
    assert_eq!(
        crate::kernel_munmap([fixed, 0x4000, 0, 0, 0, 0], &mut state),
        0
    );
    assert_eq!(
        crate::kernel_munmap([hinted, 0x4000, 0, 0, 0, 0], &mut state),
        0
    );
}

#[cfg(any(target_os = "macos", windows))]
#[test]
fn gen5_direct_memory_v2_and_checked_release_preserve_their_distinct_abi() {
    let mut state = ProbeDispatchState::default();
    assert_eq!(
        crate::kernel_checked_release_direct_memory([0, 0x4000, 0, 0, 0, 0], &mut state),
        sce_kernel_error(SCE_KERNEL_ERROR_NOT_FOUND)
    );
    let mut direct = u64::MAX;
    assert_eq!(
        allocate_direct_memory(
            [
                0,
                0x4000,
                0x4000,
                0x4000,
                3,
                (&raw mut direct).addr() as u64,
            ],
            &mut state,
        ),
        0
    );
    let mut mapped = 0_u64;
    assert_eq!(
        crate::map_direct_memory_2(
            [(&raw mut mapped).addr() as u64, 0x4000, 13, 0x33, 0, direct,],
            0x4000,
            &mut state,
        ),
        0
    );
    let mapping = state
        .kernel_memory
        .as_ref()
        .unwrap()
        .query(remu_core::GuestAddress::new(mapped), false)
        .unwrap();
    assert_eq!(mapping.memory_type, 13);
    assert_eq!(
        crate::kernel_checked_release_direct_memory([direct, 0x4000, 0, 0, 0, 0], &mut state),
        0
    );
    assert!(
        state
            .kernel_memory
            .as_ref()
            .unwrap()
            .query(remu_core::GuestAddress::new(mapped), false)
            .is_some(),
        "direct-memory release preserves aliases until explicit munmap"
    );
    assert_eq!(
        crate::kernel_checked_release_direct_memory([direct, 0x4000, 0, 0, 0, 0], &mut state),
        sce_kernel_error(SCE_KERNEL_ERROR_NOT_FOUND)
    );
    assert_eq!(
        crate::kernel_release_direct_memory([direct, 0x4000, 0, 0, 0, 0], &mut state),
        0
    );
    assert_eq!(
        crate::kernel_munmap([mapped, 0x4000, 0, 0, 0, 0], &mut state),
        0
    );
}
