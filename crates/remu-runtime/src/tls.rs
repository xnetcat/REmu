use remu_core::GuestAddress;
use remu_loader::LoadedExecutable;
use remu_memory::{GUEST_EXECUTE, GUEST_READ, GUEST_WRITE, GuestImage};

use crate::{
    GuestTls, LinkError, MACOS_GUEST_PAGE_ALIGNMENT, PT_TLS, add_address,
    current_guest_thread_pointer, runtime_address_at, subtract_address, write_u64,
};

pub(crate) const GUEST_TLS_PREFIX_SIZE: u64 = 0x20_000;

pub(crate) const GUEST_TLS_TCB_SIZE: u64 = 0x10_000;

const TLS_HELPER_SIZE: u64 = 96;

const TLS_HELPER_SIZE_USIZE: usize = 96;

const TLS_HELPER_COUNT: u64 = 16;

pub(crate) const TLS_HELPER_REGION_SIZE: u64 = TLS_HELPER_SIZE * TLS_HELPER_COUNT;

const TLS_HELPER_REGION_SIZE_USIZE: usize = 16 * TLS_HELPER_SIZE_USIZE;

#[derive(Clone, Copy)]
pub(crate) struct TlsImageInput<'a> {
    pub(crate) executable: &'a LoadedExecutable,
    pub(crate) base: GuestAddress,
    pub(crate) module_id: u64,
}

pub(crate) fn install_tls_support(
    memory: &mut GuestImage,
    images: &[TlsImageInput<'_>],
) -> Result<GuestTls, LinkError> {
    // Keep helpers beside the image so every rewritten load can use a five-byte rel32 call.
    let helper_base =
        memory.allocate_staging(TLS_HELPER_REGION_SIZE, 16, GUEST_READ | GUEST_EXECUTE)?;
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
    let patched_loads = patch_tls_loads(memory, helper_base)?;
    Ok(GuestTls {
        thread_pointer,
        static_block_size,
        patched_loads,
    })
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
    let saved_registers = [0_u8, 1, 2, 3, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15];
    let dispatcher = current_guest_thread_pointer as *const () as usize as u64;
    for destination in 0_u8..16 {
        let offset = usize::from(destination) * TLS_HELPER_SIZE_USIZE;
        let slot = &mut helpers[offset..offset + TLS_HELPER_SIZE_USIZE];
        let mut cursor = 0;
        slot[cursor] = 0x9c; // pushfq
        cursor += 1;
        for register in saved_registers {
            emit_push_register(slot, &mut cursor, register);
        }
        slot[cursor..cursor + 4].copy_from_slice(&[0x48, 0x83, 0xec, 0x08]);
        cursor += 4;
        slot[cursor..cursor + 2].copy_from_slice(&[0x48, 0xb8]);
        slot[cursor + 2..cursor + 10].copy_from_slice(&dispatcher.to_le_bytes());
        cursor += 10;
        slot[cursor..cursor + 2].copy_from_slice(&[0xff, 0xd0]);
        cursor += 2;
        slot[cursor..cursor + 4].copy_from_slice(&[0x48, 0x83, 0xc4, 0x08]);
        cursor += 4;
        let saved_index = saved_registers
            .iter()
            .position(|register| *register == destination)
            .unwrap_or(0);
        let saved_offset = (saved_registers.len() - 1 - saved_index) * 8;
        slot[cursor..cursor + 5].copy_from_slice(&[
            0x48,
            0x89,
            0x44,
            0x24,
            u8::try_from(saved_offset).map_err(|_| LinkError::AddressOverflow)?,
        ]);
        cursor += 5;
        for register in saved_registers.into_iter().rev() {
            emit_pop_register(slot, &mut cursor, register);
        }
        slot[cursor] = 0x9d; // popfq
        cursor += 1;
        slot[cursor] = 0xc3;
    }
    memory.write(helper_base, &helpers)?;
    Ok(())
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

fn patch_tls_loads(memory: &mut GuestImage, helper_base: GuestAddress) -> Result<usize, LinkError> {
    let executable_regions = memory
        .regions()
        .iter()
        .filter(|region| region.flags & GUEST_EXECUTE != 0 && region.file_size >= 9)
        .copied()
        .collect::<Vec<_>>();
    let mut patches = Vec::new();
    for region in executable_regions {
        let length =
            usize::try_from(region.file_size).map_err(|_| LinkError::ExecutableRegionTooLarge)?;
        let mut bytes = vec![0_u8; length];
        memory.read(region.start, &mut bytes)?;
        for offset in 0..=bytes.len() - 9 {
            let source = &bytes[offset..];
            if source[0] != 0x64 || !(0x48..=0x4f).contains(&source[1]) {
                continue;
            }
            let rex = source[1];
            if rex & 0x08 == 0 || source[2] != 0x8b {
                continue;
            }
            let mod_rm = source[3];
            if mod_rm >> 6 != 0 || mod_rm & 7 != 4 || source[4] != 0x25 {
                continue;
            }
            if source[5..9] != [0, 0, 0, 0] {
                continue;
            }
            let destination = ((mod_rm >> 3) & 7) | ((rex & 4) << 1);
            // Replacing an RSP destination would invalidate the call's return frame.
            if destination == 4 {
                continue;
            }
            let address = add_address(
                region.start,
                u64::try_from(offset).map_err(|_| LinkError::AddressOverflow)?,
            )?;
            patches.push((address, destination));
        }
    }

    for (address, destination) in &patches {
        let helper = add_address(helper_base, u64::from(*destination) * TLS_HELPER_SIZE)?;
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
        let mut replacement = [0x90_u8; 9];
        replacement[0] = 0xe8;
        replacement[1..5].copy_from_slice(&displacement.to_le_bytes());
        memory.write(*address, &replacement)?;
    }
    Ok(patches.len())
}
