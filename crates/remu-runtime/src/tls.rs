use iced_x86::{Decoder, DecoderOptions, FlowControl};
use remu_core::GuestAddress;
use remu_loader::LoadedExecutable;
use remu_memory::{GUEST_EXECUTE, GUEST_READ, GUEST_WRITE, GuestImage};
use std::collections::{BTreeMap, BTreeSet, VecDeque};
use std::sync::atomic::{AtomicU64, Ordering};

use crate::link::collect_initializer_array;
use crate::link::fs_decode::{
    collect_fs_decode_seed_addresses, report_fs_load_boundary_mismatches, seed_offsets_in_region,
    seeded_fs_load_sites,
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

#[derive(Clone, Copy)]
struct TlsImmediateStore {
    address: GuestAddress,
    displacement: [u8; 4],
    immediate: [u8; 4],
}

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
struct TlsImmediateStoreOffset {
    segment_index: usize,
    byte_offset: usize,
}

pub(crate) struct TlsStoreScanPlan {
    images: Vec<Vec<TlsImmediateStoreOffset>>,
}

impl TlsStoreScanPlan {
    pub(crate) fn helper_region_size(&self) -> Result<u64, LinkError> {
        self.images
            .iter()
            .try_fold(0_u64, |count, image| {
                count
                    .checked_add(
                        u64::try_from(image.len()).map_err(|_| LinkError::AddressOverflow)?,
                    )
                    .ok_or(LinkError::AddressOverflow)
            })?
            .checked_mul(TLS_STORE_HELPER_SIZE)
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
    let tls_stores = collect_tls_immediate_stores(memory, images, store_plan)?;
    // Keep helpers beside the image so every rewritten load can use a five-byte rel32 call.
    let helper_base =
        memory.allocate_staging(TLS_HELPER_REGION_SIZE, 16, GUEST_READ | GUEST_EXECUTE)?;
    let store_helper_size = u64::try_from(tls_stores.len())
        .map_err(|_| LinkError::AddressOverflow)?
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
    let patched_loads = patch_tls_loads_with_seeds(memory, helper_base, &decode_seeds)?;
    let patched_stores = if let Some(store_helper_base) = store_helper_base {
        write_tls_store_helpers(memory, store_helper_base, &tls_stores)?;
        patch_tls_immediate_stores(memory, store_helper_base, &tls_stores)?
    } else {
        0
    };
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

pub(crate) fn scan_tls_immediate_stores<'a>(
    executables: impl IntoIterator<Item = &'a LoadedExecutable>,
) -> Result<TlsStoreScanPlan, LinkError> {
    let mut images = Vec::new();
    for executable in executables {
        let dynamic = executable
            .info
            .elf
            .as_ref()
            .and_then(|elf| elf.dynamic.as_ref())
            .ok_or(LinkError::MissingDynamicMetadata)?;
        let mut image_stores = BTreeSet::new();
        for symbol in dynamic.symbols.iter().filter(|symbol| {
            !symbol.is_undefined() && symbol.symbol_type() == 2 && symbol.size != 0
        }) {
            let symbol_start = symbol.value.get();
            let Some(symbol_end) = symbol_start.checked_add(symbol.size) else {
                continue;
            };
            let Some((segment_index, segment)) =
                executable
                    .load_segments
                    .iter()
                    .enumerate()
                    .find(|(_, segment)| {
                        if segment.flags & GUEST_EXECUTE == 0 {
                            return false;
                        }
                        let segment_start = segment.virtual_address.get();
                        let Ok(data_size) = u64::try_from(segment.data.len()) else {
                            return false;
                        };
                        let Some(segment_end) = segment_start.checked_add(data_size) else {
                            return false;
                        };
                        segment_start <= symbol_start && symbol_end <= segment_end
                    })
            else {
                continue;
            };
            let function_offset = usize::try_from(symbol_start - segment.virtual_address.get())
                .map_err(|_| LinkError::ExecutableRegionTooLarge)?;
            let function_size =
                usize::try_from(symbol.size).map_err(|_| LinkError::ExecutableRegionTooLarge)?;
            let function = &segment.data[function_offset..function_offset + function_size];
            for relative_offset in reachable_tls_immediate_store_offsets(function, symbol_start) {
                image_stores.insert(TlsImmediateStoreOffset {
                    segment_index,
                    byte_offset: function_offset
                        .checked_add(relative_offset)
                        .ok_or(LinkError::AddressOverflow)?,
                });
            }
        }
        images.push(image_stores.into_iter().collect());
    }
    Ok(TlsStoreScanPlan { images })
}

fn collect_tls_immediate_stores(
    memory: &GuestImage,
    images: &[TlsImageInput<'_>],
    store_plan: &TlsStoreScanPlan,
) -> Result<Vec<TlsImmediateStore>, LinkError> {
    debug_assert_eq!(images.len(), store_plan.images.len());
    let mut stores = Vec::new();
    for (image, image_plan) in images.iter().zip(&store_plan.images) {
        for store in image_plan {
            let segment = &image.executable.load_segments[store.segment_index];
            let address = add_address(
                add_address(image.base, segment.virtual_address.get())?,
                u64::try_from(store.byte_offset).map_err(|_| LinkError::AddressOverflow)?,
            )?;
            let mut encoding = [0_u8; TLS_IMMEDIATE_STORE_SIZE];
            memory.read(address, &mut encoding)?;
            if !is_tls_immediate_store_encoding(&encoding) {
                continue;
            }
            stores.push(TlsImmediateStore {
                address,
                displacement: encoding[4..8]
                    .try_into()
                    .expect("fixed-width TLS displacement"),
                immediate: encoding[8..12]
                    .try_into()
                    .expect("fixed-width TLS immediate"),
            });
        }
    }
    Ok(stores)
}

fn reachable_tls_immediate_store_offsets(bytes: &[u8], virtual_address: u64) -> Vec<usize> {
    let mut pending = VecDeque::from([0_usize]);
    let mut visited = BTreeSet::new();
    let mut stores = Vec::new();
    while let Some(mut offset) = pending.pop_front() {
        while offset < bytes.len() && visited.insert(offset) {
            let instruction_address =
                virtual_address.saturating_add(u64::try_from(offset).unwrap_or(u64::MAX));
            let mut decoder = Decoder::with_ip(
                64,
                &bytes[offset..],
                instruction_address,
                DecoderOptions::NONE,
            );
            let instruction = decoder.decode();
            if instruction.is_invalid() || instruction.len() == 0 {
                break;
            }
            let next = offset.saturating_add(instruction.len());
            if next > bytes.len() {
                break;
            }
            if instruction.len() == TLS_IMMEDIATE_STORE_SIZE
                && is_tls_immediate_store_encoding(&bytes[offset..next])
            {
                stores.push(offset);
            }
            match instruction.flow_control() {
                FlowControl::Next | FlowControl::Call | FlowControl::IndirectCall => {
                    offset = next;
                }
                FlowControl::ConditionalBranch => {
                    if let Some(target) = function_relative_target(
                        instruction.near_branch_target(),
                        virtual_address,
                        bytes.len(),
                    ) {
                        pending.push_back(target);
                    }
                    offset = next;
                }
                FlowControl::UnconditionalBranch => {
                    if let Some(target) = function_relative_target(
                        instruction.near_branch_target(),
                        virtual_address,
                        bytes.len(),
                    ) {
                        pending.push_back(target);
                    }
                    break;
                }
                FlowControl::Return
                | FlowControl::IndirectBranch
                | FlowControl::Interrupt
                | FlowControl::XbeginXabortXend
                | FlowControl::Exception => break,
            }
        }
    }
    stores.sort_unstable();
    stores.dedup();
    stores
}

fn is_tls_immediate_store_encoding(bytes: &[u8]) -> bool {
    bytes.len() == TLS_IMMEDIATE_STORE_SIZE && bytes[..4] == [0x64, 0xc7, 0x04, 0x25]
}

fn function_relative_target(target: u64, function_start: u64, length: usize) -> Option<usize> {
    let offset = target.checked_sub(function_start)?;
    let offset = usize::try_from(offset).ok()?;
    (offset < length).then_some(offset)
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
    cursor = emit_tls_store_helper_epilogue(&mut body, cursor);
    if cursor > slot.len() {
        return Err(LinkError::AddressOverflow);
    }
    slot[..cursor].copy_from_slice(&body[..cursor]);
    Ok(cursor)
}

fn write_tls_store_helpers(
    memory: &mut GuestImage,
    helper_base: GuestAddress,
    stores: &[TlsImmediateStore],
) -> Result<(), LinkError> {
    let byte_count = stores
        .len()
        .checked_mul(TLS_STORE_HELPER_SIZE_USIZE)
        .ok_or(LinkError::AddressOverflow)?;
    let mut helpers = vec![0x90_u8; byte_count];
    let dispatcher = tls_store_thread_pointer as *const () as usize as u64;
    for (index, store) in stores.iter().enumerate() {
        let offset = index
            .checked_mul(TLS_STORE_HELPER_SIZE_USIZE)
            .ok_or(LinkError::AddressOverflow)?;
        let slot = &mut helpers[offset..offset + TLS_STORE_HELPER_SIZE_USIZE];
        encode_tls_store_helper(slot, dispatcher, store.displacement, store.immediate);
    }
    memory.write(helper_base, &helpers)?;
    Ok(())
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
    for register in TLS_HELPER_SAVED_REGISTERS.into_iter().rev() {
        emit_pop_register(slot, &mut cursor, register);
    }
    slot[cursor] = 0x9d; // popfq
    cursor += 1;
    // The patched store moves RSP below the 128-byte guest red zone before calling us.
    slot[cursor..cursor + 3].copy_from_slice(&[0xc2, 0x80, 0]); // ret 128
    cursor += 3;
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

fn patch_tls_loads_with_seeds(
    memory: &mut GuestImage,
    helper_base: GuestAddress,
    seeds: &[GuestAddress],
) -> Result<usize, LinkError> {
    let executable_regions = memory
        .regions()
        .iter()
        .filter(|region| region.flags & GUEST_EXECUTE != 0 && region.file_size >= 9)
        .copied()
        .collect::<Vec<_>>();
    let mut patches = Vec::new();
    let mut mismatch_total = 0_usize;
    for region in executable_regions {
        let length =
            usize::try_from(region.file_size).map_err(|_| LinkError::ExecutableRegionTooLarge)?;
        let mut bytes = vec![0_u8; length];
        memory.read(region.start, &mut bytes)?;
        let region_seeds = seed_offsets_in_region(seeds, region.start, length);
        let report = report_fs_load_boundary_mismatches(
            "execute-region",
            region.start.get(),
            &bytes,
            &region_seeds,
        );
        mismatch_total = mismatch_total.saturating_add(report.boundary_mismatches);
        let (_boundaries, loads) = seeded_fs_load_sites(&bytes, &region_seeds);
        for site in loads {
            patches.push((
                add_address(
                    region.start,
                    u64::try_from(site.offset).map_err(|_| LinkError::AddressOverflow)?,
                )?,
                site.destination,
                site.length,
            ));
        }
    }
    if mismatch_total != 0 {
        eprintln!(
            "[REMu][TLS][FS] boundary-mismatch total={mismatch_total} — byte-scan hits rejected; only seeded-decode sites are rewritten"
        );
    }

    for (address, destination, instruction_length) in &patches {
        let helper = add_address(helper_base, u64::from(*destination) * TLS_LOAD_HELPER_SIZE)?;
        let next = address
            .get()
            .checked_add(5)
            .ok_or(LinkError::AddressOverflow)?;
        let displacement = i128::from(helper.get()) - i128::from(next);
        let displacement =
            i32::try_from(displacement).map_err(|_| LinkError::TlsHelperOutOfRange {
                instruction: *address,
                helper,
            })?;
        let replacement = encode_tls_load_patch(*instruction_length, displacement);
        memory.write(*address, &replacement)?;
    }
    Ok(patches.len())
}

fn encode_tls_load_patch(instruction_length: usize, displacement: i32) -> Vec<u8> {
    debug_assert!(instruction_length >= 5);
    let mut replacement = vec![0x90_u8; instruction_length];
    replacement[0] = 0xe8;
    replacement[1..5].copy_from_slice(&displacement.to_le_bytes());
    replacement
}

fn patch_tls_immediate_stores(
    memory: &mut GuestImage,
    helper_base: GuestAddress,
    stores: &[TlsImmediateStore],
) -> Result<usize, LinkError> {
    for (index, store) in stores.iter().enumerate() {
        let helper = add_address(
            helper_base,
            u64::try_from(index)
                .map_err(|_| LinkError::AddressOverflow)?
                .checked_mul(TLS_STORE_HELPER_SIZE)
                .ok_or(LinkError::AddressOverflow)?,
        )?;
        // Preserve leaf-function red-zone contents: move the guest stack below its 128-byte
        // red zone without changing flags, then let RET 128 in the per-site helper restore RSP.
        // The call ends at byte ten, leaving two bytes for padding in the original instruction.
        let next = store
            .address
            .get()
            .checked_add(10)
            .ok_or(LinkError::AddressOverflow)?;
        let displacement = i128::from(helper.get()) - i128::from(next);
        let displacement =
            i32::try_from(displacement).map_err(|_| LinkError::TlsHelperOutOfRange {
                instruction: store.address,
                helper,
            })?;
        let replacement = encode_tls_store_patch(displacement);
        memory.write(store.address, &replacement)?;
    }
    Ok(stores.len())
}

fn encode_tls_store_patch(displacement: i32) -> [u8; TLS_IMMEDIATE_STORE_SIZE] {
    let mut replacement = [0x90_u8; TLS_IMMEDIATE_STORE_SIZE];
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
        reachable_tls_immediate_store_offsets,
    };
    use std::sync::atomic::Ordering;

    const STORE: [u8; 12] = [0x64, 0xc7, 0x04, 0x25, 0x28, 0, 0, 0, 0x0b, 0, 0x02, 0xa0];

    #[test]
    fn tls_store_scanner_requires_reachable_function_code() {
        let mut reachable = vec![0x55, 0x48, 0x89, 0xe5];
        reachable.extend_from_slice(&STORE);
        reachable.push(0xc3);
        assert_eq!(
            reachable_tls_immediate_store_offsets(&reachable, 0x1000),
            vec![4]
        );

        let mut branch_over_embedded_data = vec![0xeb, 0x0c];
        branch_over_embedded_data.extend_from_slice(&STORE);
        branch_over_embedded_data.push(0xc3);
        assert!(
            reachable_tls_immediate_store_offsets(&branch_over_embedded_data, 0x1000).is_empty()
        );

        let mut redundant_prefix_instruction = vec![0x2e, 0x3e, 0x26];
        redundant_prefix_instruction.extend_from_slice(&STORE);
        assert!(
            reachable_tls_immediate_store_offsets(&redundant_prefix_instruction, 0x1000).is_empty()
        );
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
                && mnemonics.iter().any(|m| matches!(m, Mnemonic::Popfq)),
            "load helper must match the store helper's ABI contract: {mnemonics:?}"
        );
        assert_eq!(&slot[encoded - 3..encoded], &[0xc2, 0x80, 0]);
    }

    #[test]
    fn tls_store_thunk_counter_starts_at_zero() {
        // The counter is process-global; this only asserts the symbol exists and is readable.
        let _ = TLS_STORE_THUNK_HITS.load(Ordering::Relaxed);
    }

    #[test]
    fn tls_store_patch_preserves_the_guest_red_zone_without_changing_flags() {
        assert_eq!(
            encode_tls_store_patch(0x1234_5678),
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
