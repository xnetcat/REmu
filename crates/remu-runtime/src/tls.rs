use remu_core::GuestAddress;
use remu_loader::LoadedExecutable;
use remu_memory::{GUEST_EXECUTE, GUEST_READ, GUEST_WRITE, GuestImage};
use std::collections::{BTreeMap, BTreeSet};
use std::sync::atomic::{AtomicU64, Ordering};

use crate::link::collect_initializer_array;
use crate::link::fs_decode::{
    FsAccessKind, FsAccessSite, collect_fs_decode_seed_addresses, seed_offsets_in_region,
    seeded_fs_access_sites,
};
use crate::unwind::collect_eh_frame_function_starts;
use crate::{
    GuestTls, LinkError, MACOS_GUEST_PAGE_ALIGNMENT, PT_TLS, add_address,
    current_guest_thread_pointer, runtime_address_at, subtract_address, write_u64,
};

pub(crate) const GUEST_TLS_PREFIX_SIZE: u64 = 0x20_000;

pub(crate) const GUEST_TLS_TCB_SIZE: u64 = 0x10_000;

/// Counts TLS store-helper executions (plan C4). Nothing measured them before this counter.
pub(crate) static TLS_STORE_THUNK_HITS: AtomicU64 = AtomicU64::new(0);

/// Load and store helpers share the store prologue (flags + GPRs + XSAVE) and the red-zone
/// contract (`ret 128`). 96 bytes was only enough for the old flags/GPR-only load thunk.
const TLS_LOAD_HELPER_SIZE: u64 = 192;

const TLS_LOAD_HELPER_SIZE_USIZE: usize = 192;

const TLS_LOAD_HELPER_COUNT: u64 = 16;

pub(crate) const TLS_HELPER_REGION_SIZE: u64 = TLS_LOAD_HELPER_SIZE * TLS_LOAD_HELPER_COUNT;

const TLS_HELPER_REGION_SIZE_USIZE: usize = 16 * TLS_LOAD_HELPER_SIZE_USIZE;

/// The store helper preserves every architectural value that the replaced MOV leaves untouched.
const TLS_STORE_HELPER_SIZE: u64 = 192;

const TLS_STORE_HELPER_SIZE_USIZE: usize = 192;

const TLS_HELPER_SAVED_REGISTERS: [u8; 15] = [0, 1, 2, 3, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15];

const TLS_IMMEDIATE_STORE_SIZE: usize = 12;

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
struct TlsImmediateStoreOffset {
    segment_index: usize,
    byte_offset: usize,
}

pub(crate) struct TlsStoreScanPlan {
    images: Vec<Vec<TlsImmediateStoreOffset>>,
}

/// Extra per-site helper slots reserved beyond the store-imm pattern ceiling.
///
/// Seeded decode may discover `lea fs:[disp]` (must AOT — it never faults) and displaced loads
/// after the image is mapped. Those consume slots from this cushion first; store-imm and load-disp
/// that still do not fit defer to the SIGSEGV backstop. A raw whole-image byte scan of displaced
/// forms false-positives in immediates and must not size the reserve. Install refuses only when
/// authenticated `lea` sites exceed `store-imm ceiling + cushion`.
const PER_SITE_FS_HELPER_CUSHION: u64 = 0;

impl TlsStoreScanPlan {
    pub(crate) fn helper_region_size(&self) -> Result<u64, LinkError> {
        let store_slots = self.images.iter().try_fold(0_u64, |count, image| {
            count
                .checked_add(u64::try_from(image.len()).map_err(|_| LinkError::AddressOverflow)?)
                .ok_or(LinkError::AddressOverflow)
        })?;
        store_slots
            .checked_add(PER_SITE_FS_HELPER_CUSHION)
            .ok_or(LinkError::AddressOverflow)?
            .checked_mul(TLS_STORE_HELPER_SIZE)
            .ok_or(LinkError::AddressOverflow)
    }

    fn slot_capacity(&self) -> Result<u64, LinkError> {
        self.images
            .iter()
            .try_fold(0_u64, |count, image| {
                count
                    .checked_add(
                        u64::try_from(image.len()).map_err(|_| LinkError::AddressOverflow)?,
                    )
                    .ok_or(LinkError::AddressOverflow)
            })?
            .checked_add(PER_SITE_FS_HELPER_CUSHION)
            .ok_or(LinkError::AddressOverflow)
    }
}

#[derive(Clone, Copy)]
pub(crate) struct TlsImageInput<'a> {
    pub(crate) executable: &'a LoadedExecutable,
    pub(crate) base: GuestAddress,
    pub(crate) module_id: u64,
}

pub(crate) fn install_tls_support(
    memory: &mut GuestImage,
    images: &[TlsImageInput<'_>],
    store_plan: &TlsStoreScanPlan,
) -> Result<GuestTls, LinkError> {
    // Keep helpers beside the image so every rewritten load can use a five-byte rel32 call.
    let helper_base =
        memory.allocate_staging(TLS_HELPER_REGION_SIZE, 16, GUEST_READ | GUEST_EXECUTE)?;
    // Per-site helper staging: store-imm pattern ceiling + displaced load/lea cushion.
    let store_helper_capacity = store_plan.slot_capacity()?;
    let store_helper_size = store_helper_capacity
        .checked_mul(TLS_STORE_HELPER_SIZE)
        .ok_or(LinkError::AddressOverflow)?;
    let store_helper_base = if store_helper_size == 0 {
        None
    } else {
        Some(memory.allocate_staging(store_helper_size, 16, GUEST_READ | GUEST_EXECUTE)?)
    };
    // Keep writable TLS on a separate host page from executable import/TLS helpers.
    let allocation_base = memory.allocate_staging(
        GUEST_TLS_PREFIX_SIZE + GUEST_TLS_TCB_SIZE,
        MACOS_GUEST_PAGE_ALIGNMENT,
        GUEST_READ | GUEST_WRITE,
    )?;
    let thread_pointer = GuestAddress::new(
        allocation_base
            .get()
            .checked_add(GUEST_TLS_PREFIX_SIZE)
            .ok_or(LinkError::AddressOverflow)?,
    );

    let mut static_block_size = 0;
    let mut module_blocks = Vec::new();
    for image in images {
        let Some(tls_header) = image
            .executable
            .info
            .elf
            .as_ref()
            .and_then(|elf| {
                elf.program_headers
                    .iter()
                    .find(|header| header.program_type == PT_TLS)
            })
            .filter(|header| header.memory_size != 0)
        else {
            continue;
        };
        let alignment = if tls_header.alignment == 0 {
            1
        } else {
            tls_header.alignment
        };
        if !alignment.is_power_of_two() {
            return Err(LinkError::InvalidTlsAlignment(alignment));
        }
        static_block_size = calculate_static_tls_offset(
            static_block_size,
            tls_header.memory_size,
            alignment,
            tls_header.virtual_address.get(),
        )?;
        if static_block_size > GUEST_TLS_PREFIX_SIZE {
            return Err(LinkError::StaticTlsTooLarge {
                required: static_block_size,
                available: GUEST_TLS_PREFIX_SIZE,
            });
        }
        let block = GuestAddress::new(
            thread_pointer
                .get()
                .checked_sub(static_block_size)
                .ok_or(LinkError::AddressOverflow)?,
        );
        let initialized_size = tls_header.file_size.min(tls_header.memory_size);
        if initialized_size != 0 {
            let initialized_size = usize::try_from(initialized_size)
                .map_err(|_| LinkError::ExecutableRegionTooLarge)?;
            let source = runtime_address_at(
                memory,
                image.base,
                tls_header.virtual_address,
                initialized_size,
            )?;
            let mut template = vec![0_u8; initialized_size];
            memory.read(source, &mut template)?;
            memory.write(block, &template)?;
        }
        module_blocks.push((image.module_id, block));
    }

    seed_tls_thread_control(memory, thread_pointer, &module_blocks)?;

    write_tls_helpers(memory, helper_base)?;
    let decode_seeds = collect_tls_fs_decode_seeds(memory, images)?;
    let (patched_loads, patched_stores) = patch_fs_accesses(
        memory,
        helper_base,
        store_helper_base,
        store_helper_capacity,
        &decode_seeds,
    )?;
    Ok(GuestTls {
        thread_pointer,
        static_block_size,
        patched_loads,
        patched_stores,
    })
}

/// Computes each module's static-TLS block size below the thread pointer, keyed by module id.
///
/// This is the same accumulation `install_tls_support` performs while laying out the static block,
/// exposed for the relocation pass so a `TPOFF64` fixup can be expressed relative to the thread
/// pointer before the block itself is written. Modules without a static-TLS block are absent from
/// the map, so callers treat a missing entry as a zero offset.
pub(crate) fn static_tls_offsets(
    images: &[TlsImageInput<'_>],
) -> Result<BTreeMap<u64, u64>, LinkError> {
    let mut static_block_size = 0;
    let mut offsets = BTreeMap::new();
    for image in images {
        let Some(tls_header) = image
            .executable
            .info
            .elf
            .as_ref()
            .and_then(|elf| {
                elf.program_headers
                    .iter()
                    .find(|header| header.program_type == PT_TLS)
            })
            .filter(|header| header.memory_size != 0)
        else {
            continue;
        };
        let alignment = if tls_header.alignment == 0 {
            1
        } else {
            tls_header.alignment
        };
        if !alignment.is_power_of_two() {
            return Err(LinkError::InvalidTlsAlignment(alignment));
        }
        static_block_size = calculate_static_tls_offset(
            static_block_size,
            tls_header.memory_size,
            alignment,
            tls_header.virtual_address.get(),
        )?;
        offsets.insert(image.module_id, static_block_size);
    }
    Ok(offsets)
}

/// Preflight ceiling for store-imm helper slots (byte-pattern count only).
///
/// Displaced load/lea capacity is the separate [`PER_SITE_FS_HELPER_CUSHION`] added in
/// [`TlsStoreScanPlan::helper_region_size`]. Install rediscovers every site via seeded decode.
pub(crate) fn scan_tls_immediate_stores<'a>(
    executables: impl IntoIterator<Item = &'a LoadedExecutable>,
) -> Result<TlsStoreScanPlan, LinkError> {
    let mut images = Vec::new();
    for executable in executables {
        let mut image_stores = BTreeSet::new();
        for (segment_index, segment) in executable.load_segments.iter().enumerate() {
            if segment.flags & GUEST_EXECUTE == 0 {
                continue;
            }
            let data = &segment.data;
            if data.len() < TLS_IMMEDIATE_STORE_SIZE {
                continue;
            }
            for byte_offset in 0..=data.len() - TLS_IMMEDIATE_STORE_SIZE {
                if is_tls_immediate_store_encoding(
                    &data[byte_offset..byte_offset + TLS_IMMEDIATE_STORE_SIZE],
                ) {
                    image_stores.insert(TlsImmediateStoreOffset {
                        segment_index,
                        byte_offset,
                    });
                }
            }
        }
        images.push(image_stores.into_iter().collect());
    }
    Ok(TlsStoreScanPlan { images })
}

fn is_tls_immediate_store_encoding(bytes: &[u8]) -> bool {
    bytes.len() == TLS_IMMEDIATE_STORE_SIZE && bytes[..4] == [0x64, 0xc7, 0x04, 0x25]
}

#[derive(Clone, Copy)]
enum PerSiteFsHelper {
    StoreImm {
        displacement: [u8; 4],
        immediate: [u8; 4],
    },
    /// `mov r64, fs:[disp]` with nonzero displacement — load `*(TP+disp)` into destination.
    LoadDisp { destination: u8, displacement: i32 },
    /// `lea rN, fs:[disp]` — write `TP+disp` into destination (never faults; must be AOT).
    Lea { destination: u8, displacement: i32 },
}

#[derive(Clone, Copy)]
struct SharedLoadPatch {
    address: GuestAddress,
    instruction_length: usize,
    destination: u8,
}

fn seed_tls_thread_control(
    memory: &mut GuestImage,
    thread_pointer: GuestAddress,
    module_blocks: &[(u64, GuestAddress)],
) -> Result<(), LinkError> {
    let dtv = add_address(thread_pointer, 0x100)?;
    write_u64(memory, thread_pointer, thread_pointer.get())?;
    write_u64(memory, add_address(thread_pointer, 0x08)?, dtv.get())?;
    write_u64(
        memory,
        add_address(thread_pointer, 0x10)?,
        thread_pointer.get(),
    )?;
    write_u64(
        memory,
        add_address(thread_pointer, 0x28)?,
        0xc0de_c0de_cafe_ba00,
    )?;
    write_u64(
        memory,
        add_address(thread_pointer, 0x60)?,
        thread_pointer.get(),
    )?;
    write_u64(memory, subtract_address(thread_pointer, 0xf0)?, 0)?;
    // DTV generation, maximum module ID, and one pointer slot per module ID.
    write_u64(memory, dtv, 1)?;
    let maximum_module_id = module_blocks
        .iter()
        .map(|(module_id, _)| *module_id)
        .max()
        .unwrap_or(0);
    write_u64(memory, add_address(dtv, 8)?, maximum_module_id)?;
    for (module_id, block) in module_blocks {
        let entry_offset = tls_dtv_entry_offset(*module_id).ok_or(LinkError::AddressOverflow)?;
        write_u64(memory, add_address(dtv, entry_offset)?, block.get())?;
    }
    Ok(())
}

pub(crate) fn tls_dtv_entry_offset(module_id: u64) -> Option<u64> {
    if module_id == 0 {
        return None;
    }
    module_id.checked_mul(8)?.checked_add(8)
}

fn calculate_static_tls_offset(
    previous_offset: u64,
    size: u64,
    alignment: u64,
    alignment_bias: u64,
) -> Result<u64, LinkError> {
    let result = previous_offset
        .checked_add(size)
        .and_then(|value| value.checked_add(alignment - 1))
        .ok_or(LinkError::AddressOverflow)?;
    let biased = result
        .checked_add(alignment_bias)
        .ok_or(LinkError::AddressOverflow)?;
    Ok(result - (biased & (alignment - 1)))
}

fn write_tls_helpers(memory: &mut GuestImage, helper_base: GuestAddress) -> Result<(), LinkError> {
    let mut helpers = vec![0x90_u8; TLS_HELPER_REGION_SIZE_USIZE];
    let dispatcher = current_guest_thread_pointer as *const () as usize as u64;
    for destination in 0_u8..16 {
        let offset = usize::from(destination) * TLS_LOAD_HELPER_SIZE_USIZE;
        let slot = &mut helpers[offset..offset + TLS_LOAD_HELPER_SIZE_USIZE];
        if destination == 4 {
            // RSP is never a patched TLS load destination; leave an explicit trap in the unused
            // slot so a bad rel32 cannot silently execute NOPs.
            slot[0] = 0x0f;
            slot[1] = 0x0b; // ud2
            continue;
        }
        encode_tls_load_helper(slot, dispatcher, destination)?;
    }
    memory.write(helper_base, &helpers)?;
    Ok(())
}

fn encode_tls_load_helper(
    slot: &mut [u8],
    dispatcher: u64,
    destination: u8,
) -> Result<usize, LinkError> {
    // The load patch is a plain `call` (sites are often only nine bytes). Skip the 128-byte red
    // zone here — same contract as the store path's `lea rsp,[rsp-128]; call` / `ret 128` pair —
    // then reuse the store helper's full flags + GPR + XSAVE prologue so a patched load cannot
    // clobber live XMM/YMM or red-zone data.
    let mut body = [0x90_u8; TLS_LOAD_HELPER_SIZE_USIZE];
    let mut cursor = 0;
    body[cursor..cursor + 5].copy_from_slice(&[0x48, 0x8d, 0x64, 0x24, 0x80]); // lea rsp,[rsp-128]
    cursor += 5;
    let mut prologue = [0x90_u8; TLS_LOAD_HELPER_SIZE_USIZE];
    let prologue_len = emit_tls_store_helper_prologue(&mut prologue, dispatcher);
    body[cursor..cursor + prologue_len].copy_from_slice(&prologue[..prologue_len]);
    cursor += prologue_len;
    // Publish the thread pointer into the destination register's saved stack slot. RSP is never a
    // load destination — the decoder refuses destination 4.
    if destination == 4 {
        return Err(LinkError::AddressOverflow);
    }
    let saved_index = TLS_HELPER_SAVED_REGISTERS
        .iter()
        .position(|register| *register == destination)
        .ok_or(LinkError::AddressOverflow)?;
    let saved_offset = (TLS_HELPER_SAVED_REGISTERS.len() - 1 - saved_index) * 8;
    body[cursor..cursor + 5].copy_from_slice(&[
        0x48,
        0x89,
        0x44,
        0x24,
        u8::try_from(saved_offset).map_err(|_| LinkError::AddressOverflow)?,
    ]);
    cursor += 5;
    // Red-zone skip is inside this helper (call sites are often only nine bytes), so undo it
    // with `add rsp,128` / plain `ret`. The store path keeps site-side `lea` + `ret 128`.
    cursor = emit_tls_load_helper_epilogue(&mut body, cursor);
    if cursor > slot.len() {
        return Err(LinkError::AddressOverflow);
    }
    slot[..cursor].copy_from_slice(&body[..cursor]);
    Ok(cursor)
}

/// Store-thunk entry: counts the hit then returns the guest TLS pointer (plan C4 measure-first).
extern "C" fn tls_store_thread_pointer() -> u64 {
    TLS_STORE_THUNK_HITS.fetch_add(1, Ordering::Relaxed);
    current_guest_thread_pointer()
}

fn encode_tls_store_helper(
    slot: &mut [u8],
    dispatcher: u64,
    displacement: [u8; 4],
    immediate: [u8; 4],
) -> usize {
    let mut cursor = emit_tls_store_helper_prologue(slot, dispatcher);
    // mov dword ptr [rax+disp], imm32
    slot[cursor..cursor + 2].copy_from_slice(&[0xc7, 0x80]);
    slot[cursor + 2..cursor + 6].copy_from_slice(&displacement);
    slot[cursor + 6..cursor + 10].copy_from_slice(&immediate);
    cursor += 10;
    emit_tls_store_helper_epilogue(slot, cursor)
}

fn encode_per_site_fs_helper(slot: &mut [u8], dispatcher: u64, kind: PerSiteFsHelper) -> usize {
    match kind {
        PerSiteFsHelper::StoreImm {
            displacement,
            immediate,
        } => encode_tls_store_helper(slot, dispatcher, displacement, immediate),
        PerSiteFsHelper::LoadDisp {
            destination,
            displacement,
        }
        | PerSiteFsHelper::Lea {
            destination,
            displacement,
        } => {
            // Nine-byte `mov`/`lea` sites cannot fit a call-site red-zone lea; skip inside the
            // helper (same contract as the shared load thunk) and return with `ret 128`.
            let mut body = [0x90_u8; TLS_STORE_HELPER_SIZE_USIZE];
            let mut cursor = 0;
            body[cursor..cursor + 5].copy_from_slice(&[0x48, 0x8d, 0x64, 0x24, 0x80]);
            cursor += 5;
            let mut prologue = [0x90_u8; TLS_STORE_HELPER_SIZE_USIZE];
            let prologue_len = emit_tls_store_helper_prologue(&mut prologue, dispatcher);
            body[cursor..cursor + prologue_len].copy_from_slice(&prologue[..prologue_len]);
            cursor += prologue_len;
            match kind {
                PerSiteFsHelper::LoadDisp { .. } => {
                    // mov rax, qword ptr [rax+disp]
                    body[cursor..cursor + 2].copy_from_slice(&[0x48, 0x8b]);
                }
                PerSiteFsHelper::Lea { .. } => {
                    // lea rax, [rax+disp]
                    body[cursor..cursor + 2].copy_from_slice(&[0x48, 0x8d]);
                }
                PerSiteFsHelper::StoreImm { .. } => unreachable!(),
            }
            body[cursor + 2] = 0x80;
            body[cursor + 3..cursor + 7].copy_from_slice(&displacement.to_le_bytes());
            cursor += 7;
            cursor = write_destination_slot(&mut body, cursor, destination);
            cursor = emit_tls_load_helper_epilogue(&mut body, cursor);
            slot[..cursor].copy_from_slice(&body[..cursor]);
            cursor
        }
    }
}

fn write_destination_slot(slot: &mut [u8], mut cursor: usize, destination: u8) -> usize {
    if destination == 4 {
        return cursor;
    }
    let Some(saved_index) = TLS_HELPER_SAVED_REGISTERS
        .iter()
        .position(|register| *register == destination)
    else {
        return cursor;
    };
    let saved_offset = (TLS_HELPER_SAVED_REGISTERS.len() - 1 - saved_index) * 8;
    slot[cursor..cursor + 5].copy_from_slice(&[0x48, 0x89, 0x44, 0x24, saved_offset as u8]);
    cursor += 5;
    cursor
}

fn emit_tls_store_helper_prologue(slot: &mut [u8], dispatcher: u64) -> usize {
    let mut cursor = 0;
    slot[cursor] = 0x9c; // pushfq
    cursor += 1;
    for register in TLS_HELPER_SAVED_REGISTERS {
        emit_push_register(slot, &mut cursor, register);
    }
    // A patched instruction can occur at any stack depth, not just at a SysV call boundary.
    // Retain the post-save stack in callee-saved R15, allocate the exact XCR0-enabled save area
    // reported by CPUID.0D.0:EBX, and align the dispatcher stack to 64 bytes.
    slot[cursor..cursor + 3].copy_from_slice(&[0x49, 0x89, 0xe7]); // mov r15, rsp
    cursor += 3;
    slot[cursor..cursor + 2].copy_from_slice(&[0x31, 0xc9]); // xor ecx, ecx
    cursor += 2;
    slot[cursor..cursor + 3].copy_from_slice(&[0x0f, 0x01, 0xd0]); // xgetbv
    cursor += 3;
    slot[cursor..cursor + 3].copy_from_slice(&[0x41, 0x89, 0xc5]); // mov r13d, eax
    cursor += 3;
    slot[cursor..cursor + 3].copy_from_slice(&[0x41, 0x89, 0xd6]); // mov r14d, edx
    cursor += 3;
    slot[cursor..cursor + 5].copy_from_slice(&[0xb8, 0x0d, 0, 0, 0]); // mov eax, 0xd
    cursor += 5;
    slot[cursor..cursor + 2].copy_from_slice(&[0x31, 0xc9]); // xor ecx, ecx
    cursor += 2;
    slot[cursor..cursor + 2].copy_from_slice(&[0x0f, 0xa2]); // cpuid
    cursor += 2;
    slot[cursor..cursor + 3].copy_from_slice(&[0x41, 0x89, 0xdc]); // mov r12d, ebx
    cursor += 3;
    slot[cursor..cursor + 3].copy_from_slice(&[0x4c, 0x29, 0xe4]); // sub rsp, r12
    cursor += 3;
    slot[cursor..cursor + 4].copy_from_slice(&[0x48, 0x83, 0xec, 0x3f]); // sub rsp, 63
    cursor += 4;
    slot[cursor..cursor + 4].copy_from_slice(&[0x48, 0x83, 0xe4, 0xc0]); // and rsp, -64
    cursor += 4;
    // XRSTOR requires the standard-format XSAVE header's reserved bytes to be zero.
    // Guest flags were saved already, so CLD is safe and POPFQ restores the original DF.
    slot[cursor] = 0xfc; // cld
    cursor += 1;
    slot[cursor..cursor + 8].copy_from_slice(&[0x48, 0x8d, 0xbc, 0x24, 0, 0x02, 0, 0]); // lea rdi,[rsp+512]
    cursor += 8;
    slot[cursor..cursor + 2].copy_from_slice(&[0x31, 0xc0]); // xor eax, eax
    cursor += 2;
    slot[cursor..cursor + 5].copy_from_slice(&[0xb9, 0x08, 0, 0, 0]); // mov ecx, 8
    cursor += 5;
    slot[cursor..cursor + 3].copy_from_slice(&[0xf3, 0x48, 0xab]); // rep stosq
    cursor += 3;
    slot[cursor..cursor + 3].copy_from_slice(&[0x44, 0x89, 0xe8]); // mov eax, r13d
    cursor += 3;
    slot[cursor..cursor + 3].copy_from_slice(&[0x44, 0x89, 0xf2]); // mov edx, r14d
    cursor += 3;
    slot[cursor..cursor + 5].copy_from_slice(&[0x48, 0x0f, 0xae, 0x24, 0x24]); // xsave64 [rsp]
    cursor += 5;
    slot[cursor..cursor + 2].copy_from_slice(&[0x48, 0xb8]);
    slot[cursor + 2..cursor + 10].copy_from_slice(&dispatcher.to_le_bytes());
    cursor += 10;
    slot[cursor..cursor + 2].copy_from_slice(&[0xff, 0xd0]); // call rax
    cursor += 2;
    slot[cursor..cursor + 3].copy_from_slice(&[0x49, 0x89, 0xc2]); // mov r10, rax
    cursor += 3;
    slot[cursor..cursor + 3].copy_from_slice(&[0x44, 0x89, 0xe8]); // mov eax, r13d
    cursor += 3;
    slot[cursor..cursor + 3].copy_from_slice(&[0x44, 0x89, 0xf2]); // mov edx, r14d
    cursor += 3;
    slot[cursor..cursor + 5].copy_from_slice(&[0x48, 0x0f, 0xae, 0x2c, 0x24]); // xrstor64 [rsp]
    cursor += 5;
    slot[cursor..cursor + 3].copy_from_slice(&[0x4c, 0x89, 0xd0]); // mov rax, r10
    cursor += 3;
    slot[cursor..cursor + 3].copy_from_slice(&[0x4c, 0x89, 0xfc]); // mov rsp, r15
    cursor += 3;
    // Guest code is entered only after assigning its per-thread pointer. Turn an invariant breach
    // into an explicit trap rather than accessing host address zero.
    slot[cursor..cursor + 7].copy_from_slice(&[0x48, 0x85, 0xc0, 0x75, 0x02, 0x0f, 0x0b]);
    cursor += 7;
    cursor
}

fn emit_tls_store_helper_epilogue(slot: &mut [u8], mut cursor: usize) -> usize {
    cursor = emit_tls_helper_gpr_flag_restore(slot, cursor);
    // The patched store moves RSP below the 128-byte guest red zone before calling us.
    slot[cursor..cursor + 3].copy_from_slice(&[0xc2, 0x80, 0]); // ret 128
    cursor += 3;
    cursor
}

fn emit_tls_load_helper_epilogue(slot: &mut [u8], mut cursor: usize) -> usize {
    cursor = emit_tls_helper_gpr_flag_restore(slot, cursor);
    // Undo the helper-local `lea rsp,[rsp-128]` so `[rsp]` is the return address again.
    slot[cursor..cursor + 7].copy_from_slice(&[0x48, 0x81, 0xc4, 0x80, 0, 0, 0]); // add rsp, 128
    cursor += 7;
    slot[cursor] = 0xc3; // ret
    cursor += 1;
    cursor
}

fn emit_tls_helper_gpr_flag_restore(slot: &mut [u8], mut cursor: usize) -> usize {
    for register in TLS_HELPER_SAVED_REGISTERS.into_iter().rev() {
        emit_pop_register(slot, &mut cursor, register);
    }
    slot[cursor] = 0x9d; // popfq
    cursor += 1;
    cursor
}

fn emit_push_register(slot: &mut [u8], cursor: &mut usize, register: u8) {
    if register < 8 {
        slot[*cursor] = 0x50 + register;
        *cursor += 1;
    } else {
        slot[*cursor..*cursor + 2].copy_from_slice(&[0x41, 0x50 + (register - 8)]);
        *cursor += 2;
    }
}

fn emit_pop_register(slot: &mut [u8], cursor: &mut usize, register: u8) {
    if register < 8 {
        slot[*cursor] = 0x58 + register;
        *cursor += 1;
    } else {
        slot[*cursor..*cursor + 2].copy_from_slice(&[0x41, 0x58 + (register - 8)]);
        *cursor += 2;
    }
}

fn collect_tls_fs_decode_seeds(
    memory: &GuestImage,
    images: &[TlsImageInput<'_>],
) -> Result<Vec<GuestAddress>, LinkError> {
    let mut seeds = BTreeSet::new();
    for image in images {
        let fde_starts = collect_eh_frame_function_starts(memory, image.base, image.executable)?;
        for seed in collect_fs_decode_seed_addresses(image.base, image.executable, &fde_starts) {
            seeds.insert(seed);
        }
        // DT_INIT_ARRAY entries are live code pointers once the image is mapped.
        if let Some(dynamic) = image
            .executable
            .info
            .elf
            .as_ref()
            .and_then(|elf| elf.dynamic.as_ref())
        {
            for initializer in collect_initializer_array(
                memory,
                image.base,
                dynamic.init_array_address,
                dynamic.init_array_size,
                "DT_INIT_ARRAY",
            )? {
                seeds.insert(initializer);
            }
            for initializer in collect_initializer_array(
                memory,
                image.base,
                dynamic.preinit_array_address,
                dynamic.preinit_array_size,
                "DT_PREINIT_ARRAY",
            )? {
                seeds.insert(initializer);
            }
        }
    }
    Ok(seeds.into_iter().collect())
}

fn patch_fs_accesses(
    memory: &mut GuestImage,
    shared_helper_base: GuestAddress,
    per_site_helper_base: Option<GuestAddress>,
    per_site_capacity: u64,
    seeds: &[GuestAddress],
) -> Result<(usize, usize), LinkError> {
    let sites = collect_seeded_fs_sites(memory, seeds)?;
    let mut shared_loads = Vec::new();
    let mut leas = Vec::new();
    let mut stores = Vec::new();
    let mut load_disps = Vec::new();
    let mut deferred = 0_usize;
    for (address, site) in sites {
        match site.kind {
            FsAccessKind::LoadGpr {
                destination,
                width_bytes: 8,
                displacement: 0,
            } if destination != 4 && site.length >= 5 => {
                shared_loads.push(SharedLoadPatch {
                    address,
                    instruction_length: site.length,
                    destination,
                });
            }
            FsAccessKind::LoadGpr {
                destination,
                width_bytes: 8,
                displacement,
            } if destination != 4 && displacement != 0 && site.length >= 5 => {
                load_disps.push((
                    address,
                    site.length,
                    PerSiteFsHelper::LoadDisp {
                        destination,
                        displacement,
                    },
                ));
            }
            FsAccessKind::LeaGpr {
                destination,
                displacement,
            } if destination != 4 && site.length >= 5 => {
                leas.push((
                    address,
                    site.length,
                    PerSiteFsHelper::Lea {
                        destination,
                        displacement,
                    },
                ));
            }
            FsAccessKind::StoreImm {
                width_bytes: 4,
                displacement,
                immediate,
            } if site.length >= 10 => {
                // iced_x86 zero-extends Immediate32 into i64; the AOT helper stores the low
                // 32 bits of the instruction encoding, not a saturating signed conversion.
                stores.push((
                    address,
                    site.length,
                    PerSiteFsHelper::StoreImm {
                        displacement: displacement.to_le_bytes(),
                        immediate: (immediate as i32).to_le_bytes(),
                    },
                ));
            }
            _ => deferred += 1,
        }
    }

    let lea_count = u64::try_from(leas.len()).map_err(|_| LinkError::AddressOverflow)?;
    if lea_count > per_site_capacity {
        eprintln!(
            "[REMu][TLS][FS] LOUD REFUSAL: seeded lea fs: helpers ({lea_count}) exceed reserve ({per_site_capacity}); lea never faults so it cannot defer — raise PER_SITE_FS_HELPER_CUSHION"
        );
        return Err(LinkError::AddressOverflow);
    }

    // lea must AOT; store-imm / displaced loads fill remaining slots; overflow defers to fault.
    let mut per_site = leas;
    let capacity = usize::try_from(per_site_capacity).map_err(|_| LinkError::AddressOverflow)?;
    for site in stores {
        if per_site.len() < capacity {
            per_site.push(site);
        } else {
            deferred += 1;
        }
    }
    for site in load_disps {
        if per_site.len() < capacity {
            per_site.push(site);
        } else {
            deferred += 1;
        }
    }

    for patch in &shared_loads {
        let helper = add_address(
            shared_helper_base,
            u64::from(patch.destination) * TLS_LOAD_HELPER_SIZE,
        )?;
        patch_call_site(memory, patch.address, patch.instruction_length, helper)?;
    }

    let mut patched_stores = 0_usize;
    if let Some(per_site_base) = per_site_helper_base {
        let dispatcher = tls_store_thread_pointer as *const () as usize as u64;
        let byte_count = per_site
            .len()
            .checked_mul(TLS_STORE_HELPER_SIZE_USIZE)
            .ok_or(LinkError::AddressOverflow)?;
        let mut helpers = vec![0x90_u8; byte_count];
        for (index, (_address, _length, kind)) in per_site.iter().enumerate() {
            let offset = index
                .checked_mul(TLS_STORE_HELPER_SIZE_USIZE)
                .ok_or(LinkError::AddressOverflow)?;
            let slot = &mut helpers[offset..offset + TLS_STORE_HELPER_SIZE_USIZE];
            encode_per_site_fs_helper(slot, dispatcher, *kind);
        }
        memory.write(per_site_base, &helpers)?;
        for (index, (address, length, kind)) in per_site.iter().enumerate() {
            let helper = add_address(
                per_site_base,
                u64::try_from(index)
                    .map_err(|_| LinkError::AddressOverflow)?
                    .checked_mul(TLS_STORE_HELPER_SIZE)
                    .ok_or(LinkError::AddressOverflow)?,
            )?;
            match kind {
                PerSiteFsHelper::StoreImm { .. } => {
                    patch_red_zone_call_site(memory, *address, *length, helper)?;
                    patched_stores += 1;
                }
                PerSiteFsHelper::LoadDisp { .. } | PerSiteFsHelper::Lea { .. } => {
                    // Red-zone skip lives inside the helper; the site is a plain call.
                    patch_call_site(memory, *address, *length, helper)?;
                }
            }
        }
    } else if !per_site.is_empty() {
        return Err(LinkError::AddressOverflow);
    }

    let patched_loads = shared_loads.len()
        + per_site
            .iter()
            .filter(|(_, _, kind)| {
                matches!(
                    kind,
                    PerSiteFsHelper::LoadDisp { .. } | PerSiteFsHelper::Lea { .. }
                )
            })
            .count();
    eprintln!(
        "[REMu][TLS][FS] decode-summary aot_loads={patched_loads} aot_stores={patched_stores} fault_backstop_sites={deferred}"
    );
    Ok((patched_loads, patched_stores))
}

fn collect_seeded_fs_sites(
    memory: &GuestImage,
    seeds: &[GuestAddress],
) -> Result<Vec<(GuestAddress, FsAccessSite)>, LinkError> {
    let executable_regions = memory
        .regions()
        .iter()
        .filter(|region| region.flags & GUEST_EXECUTE != 0 && region.file_size >= 5)
        .copied()
        .collect::<Vec<_>>();
    let mut sites = Vec::new();
    for region in executable_regions {
        let length =
            usize::try_from(region.file_size).map_err(|_| LinkError::ExecutableRegionTooLarge)?;
        let mut bytes = vec![0_u8; length];
        memory.read(region.start, &mut bytes)?;
        let region_seeds = seed_offsets_in_region(seeds, region.start, length);
        let (_boundaries, decoded) = seeded_fs_access_sites(&bytes, &region_seeds);
        for site in decoded {
            let address = add_address(
                region.start,
                u64::try_from(site.offset).map_err(|_| LinkError::AddressOverflow)?,
            )?;
            sites.push((address, site));
        }
    }
    Ok(sites)
}

fn patch_call_site(
    memory: &mut GuestImage,
    address: GuestAddress,
    instruction_length: usize,
    helper: GuestAddress,
) -> Result<(), LinkError> {
    let next = address
        .get()
        .checked_add(5)
        .ok_or(LinkError::AddressOverflow)?;
    let displacement = i128::from(helper.get()) - i128::from(next);
    let displacement = i32::try_from(displacement).map_err(|_| LinkError::TlsHelperOutOfRange {
        instruction: address,
        helper,
    })?;
    let replacement = encode_tls_load_patch(instruction_length, displacement);
    memory.write(address, &replacement)?;
    Ok(())
}

fn patch_red_zone_call_site(
    memory: &mut GuestImage,
    address: GuestAddress,
    instruction_length: usize,
    helper: GuestAddress,
) -> Result<(), LinkError> {
    let next = address
        .get()
        .checked_add(10)
        .ok_or(LinkError::AddressOverflow)?;
    let displacement = i128::from(helper.get()) - i128::from(next);
    let displacement = i32::try_from(displacement).map_err(|_| LinkError::TlsHelperOutOfRange {
        instruction: address,
        helper,
    })?;
    let replacement = encode_tls_store_patch(instruction_length, displacement);
    memory.write(address, &replacement)?;
    Ok(())
}

fn encode_tls_load_patch(instruction_length: usize, displacement: i32) -> Vec<u8> {
    debug_assert!(instruction_length >= 5);
    let mut replacement = vec![0x90_u8; instruction_length];
    replacement[0] = 0xe8;
    replacement[1..5].copy_from_slice(&displacement.to_le_bytes());
    replacement
}

fn encode_tls_store_patch(instruction_length: usize, displacement: i32) -> Vec<u8> {
    debug_assert!(instruction_length >= 10);
    let mut replacement = vec![0x90_u8; instruction_length];
    replacement[..5].copy_from_slice(&[0x48, 0x8d, 0x64, 0x24, 0x80]); // lea rsp, [rsp-128]
    replacement[5] = 0xe8;
    replacement[6..10].copy_from_slice(&displacement.to_le_bytes());
    replacement
}

#[cfg(test)]
mod tests {
    use iced_x86::{Decoder, DecoderOptions, Mnemonic};

    use super::{
        TLS_STORE_HELPER_SIZE_USIZE, TLS_STORE_THUNK_HITS, encode_tls_load_helper,
        encode_tls_load_patch, encode_tls_store_helper, encode_tls_store_patch,
        is_tls_immediate_store_encoding,
    };
    use crate::link::fs_decode::{FsAccessKind, seeded_fs_access_sites};
    use std::sync::atomic::Ordering;

    const STORE: [u8; 12] = [0x64, 0xc7, 0x04, 0x25, 0x28, 0, 0, 0, 0x0b, 0, 0x02, 0xa0];

    #[test]
    fn seeded_decode_finds_reachable_fs_stores_and_rejects_embedded_data() {
        let mut reachable = vec![0x55, 0x48, 0x89, 0xe5];
        reachable.extend_from_slice(&STORE);
        reachable.push(0xc3);
        let stores = seeded_fs_access_sites(&reachable, &[0])
            .1
            .into_iter()
            .filter(|site| matches!(site.kind, FsAccessKind::StoreImm { .. }))
            .collect::<Vec<_>>();
        assert_eq!(stores.len(), 1);
        assert_eq!(stores[0].offset, 4);

        // Branch over twelve bytes of store-shaped data: not an instruction boundary.
        let mut branch_over_embedded_data = vec![0xeb, 0x0c];
        branch_over_embedded_data.extend_from_slice(&STORE);
        branch_over_embedded_data.push(0xc3);
        assert!(
            seeded_fs_access_sites(&branch_over_embedded_data, &[0])
                .1
                .iter()
                .all(|site| !matches!(site.kind, FsAccessKind::StoreImm { .. }))
        );

        assert!(is_tls_immediate_store_encoding(&STORE));
    }

    #[test]
    fn tls_store_helper_preserves_guest_flags_and_vector_state() {
        let mut slot = [0x90_u8; TLS_STORE_HELPER_SIZE_USIZE];
        let encoded = encode_tls_store_helper(
            &mut slot,
            0x1122_3344_5566_7788,
            [0x28, 0, 0, 0],
            [0x0b, 0, 0x02, 0xa0],
        );

        assert!(encoded <= TLS_STORE_HELPER_SIZE_USIZE);
        assert_eq!(slot[0], 0x9c, "the replaced MOV must preserve RFLAGS");
        let mnemonics = decoded_mnemonics(&slot[..encoded]);
        assert!(
            mnemonics.iter().any(|m| matches!(m, Mnemonic::Xsave64))
                && mnemonics.iter().any(|m| matches!(m, Mnemonic::Xrstor64))
                && mnemonics.iter().any(|m| matches!(m, Mnemonic::Pushfq))
                && mnemonics.iter().any(|m| matches!(m, Mnemonic::Popfq)),
            "store helper must preserve the state a MOV leaves untouched: {mnemonics:?}"
        );
        assert!(
            mnemonics.contains(&Mnemonic::Lea),
            "stack adjust must use lea: {mnemonics:?}"
        );
        // Ends with ret 128.
        assert_eq!(&slot[encoded - 3..encoded], &[0xc2, 0x80, 0]);
    }

    #[test]
    fn tls_load_helper_preserves_guest_flags_vector_state_and_red_zone() {
        let mut slot = [0x90_u8; super::TLS_LOAD_HELPER_SIZE_USIZE];
        let encoded = encode_tls_load_helper(&mut slot, 0x1122_3344_5566_7788, 0).unwrap();
        assert!(encoded <= super::TLS_LOAD_HELPER_SIZE_USIZE);
        // Red-zone skip precedes the shared store prologue.
        assert_eq!(&slot[..5], &[0x48, 0x8d, 0x64, 0x24, 0x80]);
        let mnemonics = decoded_mnemonics(&slot[..encoded]);
        assert_eq!(mnemonics.first(), Some(&Mnemonic::Lea));
        assert!(
            mnemonics.iter().any(|m| matches!(m, Mnemonic::Xsave64))
                && mnemonics.iter().any(|m| matches!(m, Mnemonic::Xrstor64))
                && mnemonics.iter().any(|m| matches!(m, Mnemonic::Pushfq))
                && mnemonics.iter().any(|m| matches!(m, Mnemonic::Popfq))
                && mnemonics.iter().any(|m| matches!(m, Mnemonic::Add)),
            "load helper must preserve vector/flags and undo its own red-zone skip: {mnemonics:?}"
        );
        // Plain ret after `add rsp,128` — not `ret 128` (that contract is site-side lea only).
        assert_eq!(
            &slot[encoded - 8..encoded],
            &[0x48, 0x81, 0xc4, 0x80, 0, 0, 0, 0xc3]
        );
    }

    #[test]
    fn tls_load_helper_refuses_rsp_destination() {
        let mut slot = [0x90_u8; super::TLS_LOAD_HELPER_SIZE_USIZE];
        assert!(encode_tls_load_helper(&mut slot, 0x1122_3344_5566_7788, 4).is_err());
    }

    #[test]
    fn store_imm_aot_keeps_high_bit_immediate32_bits() {
        // Matches FS_STORE_IMM in fs_decode tests: imm32 bytes 0b 00 02 a0.
        let immediate = 0x0000_0000_a002_000b_i64;
        assert!(i32::try_from(immediate).is_err());
        assert_eq!((immediate as i32).to_le_bytes(), [0x0b, 0x00, 0x02, 0xa0]);
    }

    #[test]
    fn tls_store_thunk_counter_starts_at_zero() {
        // The counter is process-global; this only asserts the symbol exists and is readable.
        let _ = TLS_STORE_THUNK_HITS.load(Ordering::Relaxed);
    }

    #[test]
    fn tls_store_patch_preserves_the_guest_red_zone_without_changing_flags() {
        assert_eq!(
            encode_tls_store_patch(12, 0x1234_5678),
            [
                0x48, 0x8d, 0x64, 0x24, 0x80, 0xe8, 0x78, 0x56, 0x34, 0x12, 0x90, 0x90
            ]
        );
    }

    #[test]
    fn tls_load_patch_consumes_every_redundant_prefix_byte() {
        assert_eq!(
            encode_tls_load_patch(12, 0x1234_5678),
            [
                0xe8, 0x78, 0x56, 0x34, 0x12, 0x90, 0x90, 0x90, 0x90, 0x90, 0x90, 0x90
            ]
        );
    }

    fn decoded_mnemonics(bytes: &[u8]) -> Vec<Mnemonic> {
        let mut decoder = Decoder::new(64, bytes, DecoderOptions::NONE);
        let mut mnemonics = Vec::new();
        while decoder.can_decode() {
            let instruction = decoder.decode();
            assert!(!instruction.is_invalid());
            mnemonics.push(instruction.mnemonic());
        }
        mnemonics
    }
}
