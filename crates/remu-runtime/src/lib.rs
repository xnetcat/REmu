// Copyright (C) 2026 REmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

use std::collections::{BTreeMap, BTreeSet};
#[cfg(target_arch = "x86_64")]
use std::ffi::CString;
use std::sync::{OnceLock, RwLock};

use remu_core::GuestAddress;
use remu_loader::{DynamicInfo, Elf64Rela, Elf64Symbol, LoadedExecutable};
use remu_memory::{GUEST_EXECUTE, GUEST_READ, GUEST_WRITE, GuestImage, MappingError};
use thiserror::Error;

const R_X86_64_64: u32 = 1;
const R_X86_64_GLOB_DAT: u32 = 6;
const R_X86_64_JUMP_SLOT: u32 = 7;
const R_X86_64_RELATIVE: u32 = 8;
const IMPORT_STUB_SIZE: u64 = 32;
const IMPORT_STUB_SIZE_USIZE: usize = 32;
const PT_TLS: u32 = 7;
const GUEST_TLS_PREFIX_SIZE: u64 = 0x20_000;
const GUEST_TLS_TCB_SIZE: u64 = 0x10_000;
const TLS_HELPER_SIZE: u64 = 16;
const TLS_HELPER_SIZE_USIZE: usize = 16;
const TLS_HELPER_COUNT: u64 = 16;
const TLS_HELPER_REGION_SIZE: u64 = TLS_HELPER_SIZE * TLS_HELPER_COUNT;
const TLS_HELPER_REGION_SIZE_USIZE: usize = 16 * TLS_HELPER_SIZE_USIZE;
const MACOS_GUEST_PAGE_ALIGNMENT: u64 = 0x4000;

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ImportStub {
    pub nid: String,
    pub address: GuestAddress,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct RelocationStats {
    pub absolute: usize,
    pub global_data: usize,
    pub jump_slots: usize,
    pub relative: usize,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct GuestTls {
    pub thread_pointer: GuestAddress,
    pub static_block_size: u64,
    pub patched_loads: usize,
}

impl RelocationStats {
    #[must_use]
    pub const fn total(self) -> usize {
        self.absolute + self.global_data + self.jump_slots + self.relative
    }
}

#[derive(Debug)]
pub struct LinkedImage {
    pub memory: GuestImage,
    pub import_stubs: Vec<ImportStub>,
    pub relocation_stats: RelocationStats,
    pub tls: GuestTls,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ProbeOutcome {
    pub return_value: i32,
    pub import_hits: usize,
}

#[derive(Debug, Error)]
pub enum LinkError {
    #[error(transparent)]
    Mapping(#[from] MappingError),
    #[error("executable does not contain an ELF image")]
    MissingElf,
    #[error("executable does not contain dynamic-linker metadata")]
    MissingDynamicMetadata,
    #[error("import stub allocation size overflow")]
    StubAllocationOverflow,
    #[error("relocation references missing symbol {index}")]
    MissingSymbol { index: u32 },
    #[error("unresolved non-weak symbol {index} has no import stub")]
    MissingImportStub { index: u32 },
    #[error("unsupported x86-64 relocation type {relocation_type} at {offset}")]
    UnsupportedRelocation {
        relocation_type: u32,
        offset: GuestAddress,
    },
    #[error("guest address arithmetic overflow")]
    AddressOverflow,
    #[error("guest TLS alignment 0x{0:x} is not a power of two")]
    InvalidTlsAlignment(u64),
    #[error(
        "guest static TLS requires 0x{required:x} bytes, exceeding the 0x{available:x} startup reservation"
    )]
    StaticTlsTooLarge { required: u64, available: u64 },
    #[error("executable region size cannot be represented on this host")]
    ExecutableRegionTooLarge,
    #[error("TLS helper at {helper} is outside rel32 range of guest instruction {instruction}")]
    TlsHelperOutOfRange {
        instruction: GuestAddress,
        helper: GuestAddress,
    },
    #[error("native guest probing requires an x86-64 process")]
    ProbeRequiresX86_64,
    #[error("process image name contains an interior NUL byte")]
    InvalidProcessImageName,
}

impl LinkedImage {
    pub fn link_main(executable: &LoadedExecutable) -> Result<Self, LinkError> {
        let elf = executable.info.elf.as_ref().ok_or(LinkError::MissingElf)?;
        let dynamic = elf
            .dynamic
            .as_ref()
            .ok_or(LinkError::MissingDynamicMetadata)?;
        let mut memory = GuestImage::map_main(executable)?;
        let (import_stubs, symbol_stubs) = create_import_stubs(&mut memory, dynamic)?;
        let relocation_stats = apply_relocations(&mut memory, dynamic, &symbol_stubs)?;
        let tls = install_tls_support(&mut memory, executable)?;
        memory.finalize_permissions()?;
        Ok(Self {
            memory,
            import_stubs,
            relocation_stats,
            tls,
        })
    }

    pub fn probe_entry(
        &self,
        process_image_name: &str,
        maximum_import_hits: usize,
    ) -> Result<ProbeOutcome, LinkError> {
        probe_entry(self, process_image_name, maximum_import_hits)
    }
}

fn install_tls_support(
    memory: &mut GuestImage,
    executable: &LoadedExecutable,
) -> Result<GuestTls, LinkError> {
    // Keep helpers beside the image so every rewritten load can use a five-byte rel32 call.
    let helper_base = memory.allocate_staging(
        TLS_HELPER_REGION_SIZE,
        TLS_HELPER_SIZE,
        GUEST_READ | GUEST_EXECUTE,
    )?;
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
    if let Some(tls_header) = executable
        .info
        .elf
        .as_ref()
        .and_then(|elf| {
            elf.program_headers
                .iter()
                .find(|header| header.program_type == PT_TLS)
        })
        .filter(|header| header.memory_size != 0)
    {
        let alignment = if tls_header.alignment == 0 {
            1
        } else {
            tls_header.alignment
        };
        if !alignment.is_power_of_two() {
            return Err(LinkError::InvalidTlsAlignment(alignment));
        }
        static_block_size = calculate_static_tls_offset(
            0,
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
            let source = runtime_address(memory, tls_header.virtual_address, initialized_size)?;
            let mut template = vec![0_u8; initialized_size];
            memory.read(source, &mut template)?;
            memory.write(block, &template)?;
        }
    }

    seed_tls_thread_control(memory, thread_pointer, static_block_size)?;

    write_tls_helpers(memory, helper_base, thread_pointer)?;
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
    static_block_size: u64,
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
    // One startup module: generation, maximum module ID, and module-1 block address.
    write_u64(memory, dtv, 1)?;
    write_u64(
        memory,
        add_address(dtv, 8)?,
        usize::from(static_block_size != 0) as u64,
    )?;
    if static_block_size != 0 {
        write_u64(
            memory,
            add_address(dtv, 16)?,
            thread_pointer.get() - static_block_size,
        )?;
    }
    Ok(())
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

fn write_tls_helpers(
    memory: &mut GuestImage,
    helper_base: GuestAddress,
    thread_pointer: GuestAddress,
) -> Result<(), LinkError> {
    let mut helpers = vec![0x90_u8; TLS_HELPER_REGION_SIZE_USIZE];
    for destination in 0_u8..16 {
        let offset = usize::from(destination) * TLS_HELPER_SIZE_USIZE;
        let slot = &mut helpers[offset..offset + TLS_HELPER_SIZE_USIZE];
        let mut cursor = 0;
        if destination != 0 {
            // Preserve RAX because the original `mov destination, fs:[0]` did not change it.
            slot[cursor] = 0x50;
            cursor += 1;
        }
        slot[cursor..cursor + 2].copy_from_slice(&[0x48, 0xb8]);
        slot[cursor + 2..cursor + 10].copy_from_slice(&thread_pointer.get().to_le_bytes());
        cursor += 10;
        if destination != 0 {
            slot[cursor] = if destination >= 8 { 0x49 } else { 0x48 };
            slot[cursor + 1] = 0x89;
            slot[cursor + 2] = 0xc0 | (destination & 7);
            cursor += 3;
            slot[cursor] = 0x58;
            cursor += 1;
        }
        slot[cursor] = 0xc3;
    }
    memory.write(helper_base, &helpers)?;
    Ok(())
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

fn add_address(address: GuestAddress, offset: u64) -> Result<GuestAddress, LinkError> {
    Ok(GuestAddress::new(
        address
            .get()
            .checked_add(offset)
            .ok_or(LinkError::AddressOverflow)?,
    ))
}

fn subtract_address(address: GuestAddress, offset: u64) -> Result<GuestAddress, LinkError> {
    Ok(GuestAddress::new(
        address
            .get()
            .checked_sub(offset)
            .ok_or(LinkError::AddressOverflow)?,
    ))
}

fn write_u64(memory: &mut GuestImage, address: GuestAddress, value: u64) -> Result<(), LinkError> {
    memory.write(address, &value.to_le_bytes())?;
    Ok(())
}

fn create_import_stubs(
    memory: &mut GuestImage,
    dynamic: &DynamicInfo,
) -> Result<(Vec<ImportStub>, BTreeMap<u32, GuestAddress>), LinkError> {
    let nids = dynamic
        .imports
        .iter()
        .filter(|import| !import.weak)
        .map(|import| extract_nid(&import.name).to_owned())
        .collect::<BTreeSet<_>>();
    if nids.is_empty() {
        return Ok((Vec::new(), BTreeMap::new()));
    }

    let stub_bytes = u64::try_from(nids.len())
        .map_err(|_| LinkError::StubAllocationOverflow)?
        .checked_mul(IMPORT_STUB_SIZE)
        .ok_or(LinkError::StubAllocationOverflow)?;
    let stub_base =
        memory.allocate_staging(stub_bytes, IMPORT_STUB_SIZE, GUEST_READ | GUEST_EXECUTE)?;
    let bytes_len = usize::try_from(stub_bytes).map_err(|_| LinkError::StubAllocationOverflow)?;
    let mut bytes = vec![0_u8; bytes_len];
    let mut import_stubs = Vec::with_capacity(nids.len());
    let mut by_nid = BTreeMap::new();
    let dispatcher = import_dispatch as *const () as usize as u64;
    for (index, nid) in nids.into_iter().enumerate() {
        let byte_offset = index * IMPORT_STUB_SIZE_USIZE;
        write_import_stub(&mut bytes[byte_offset..], index, dispatcher)?;
        let address = GuestAddress::new(
            stub_base
                .get()
                .checked_add(byte_offset as u64)
                .ok_or(LinkError::AddressOverflow)?,
        );
        by_nid.insert(nid.clone(), address);
        import_stubs.push(ImportStub { nid, address });
    }
    memory.write(stub_base, &bytes)?;

    let mut symbol_stubs = BTreeMap::new();
    for import in &dynamic.imports {
        if import.weak {
            continue;
        }
        let nid = extract_nid(&import.name);
        let address = by_nid
            .get(nid)
            .copied()
            .ok_or(LinkError::MissingImportStub {
                index: import.symbol_index,
            })?;
        symbol_stubs.insert(import.symbol_index, address);
    }
    Ok((import_stubs, symbol_stubs))
}

fn apply_relocations(
    memory: &mut GuestImage,
    dynamic: &DynamicInfo,
    symbol_stubs: &BTreeMap<u32, GuestAddress>,
) -> Result<RelocationStats, LinkError> {
    let mut stats = RelocationStats::default();
    for relocation in dynamic.relocations.iter().chain(&dynamic.jump_relocations) {
        let target = runtime_address(memory, relocation.offset, 8)?;
        let value = match relocation.relocation_type {
            R_X86_64_RELATIVE => {
                stats.relative += 1;
                memory
                    .image_base()
                    .get()
                    .wrapping_add_signed(relocation.addend)
            }
            R_X86_64_64 => {
                stats.absolute += 1;
                resolve_symbol(memory, dynamic, symbol_stubs, relocation)?
                    .wrapping_add_signed(relocation.addend)
            }
            R_X86_64_GLOB_DAT => {
                stats.global_data += 1;
                resolve_symbol(memory, dynamic, symbol_stubs, relocation)?
            }
            R_X86_64_JUMP_SLOT => {
                stats.jump_slots += 1;
                resolve_symbol(memory, dynamic, symbol_stubs, relocation)?
            }
            relocation_type => {
                return Err(LinkError::UnsupportedRelocation {
                    relocation_type,
                    offset: relocation.offset,
                });
            }
        };
        memory.write(target, &value.to_le_bytes())?;
    }
    Ok(stats)
}

fn resolve_symbol(
    memory: &GuestImage,
    dynamic: &DynamicInfo,
    symbol_stubs: &BTreeMap<u32, GuestAddress>,
    relocation: &Elf64Rela,
) -> Result<u64, LinkError> {
    let index = usize::try_from(relocation.symbol_index).map_err(|_| LinkError::MissingSymbol {
        index: relocation.symbol_index,
    })?;
    let symbol = dynamic.symbols.get(index).ok_or(LinkError::MissingSymbol {
        index: relocation.symbol_index,
    })?;
    if symbol.is_undefined() {
        if symbol.is_weak() {
            return Ok(0);
        }
        return symbol_stubs
            .get(&relocation.symbol_index)
            .map(|address| address.get())
            .ok_or(LinkError::MissingImportStub {
                index: relocation.symbol_index,
            });
    }
    defined_symbol_address(memory, symbol)
}

fn defined_symbol_address(memory: &GuestImage, symbol: &Elf64Symbol) -> Result<u64, LinkError> {
    if memory.contains(symbol.value, 1) {
        return Ok(symbol.value.get());
    }
    memory
        .image_base()
        .get()
        .checked_add(symbol.value.get())
        .ok_or(LinkError::AddressOverflow)
}

fn runtime_address(
    memory: &GuestImage,
    value: GuestAddress,
    length: usize,
) -> Result<GuestAddress, LinkError> {
    if memory.contains(value, length) {
        return Ok(value);
    }
    Ok(GuestAddress::new(
        memory
            .image_base()
            .get()
            .checked_add(value.get())
            .ok_or(LinkError::AddressOverflow)?,
    ))
}

fn extract_nid(symbol_name: &str) -> &str {
    symbol_name
        .split_once('#')
        .map_or(symbol_name, |(nid, _)| nid)
}

fn write_import_stub(slot: &mut [u8], index: usize, dispatcher: u64) -> Result<(), LinkError> {
    let index = u32::try_from(index).map_err(|_| LinkError::StubAllocationOverflow)?;
    // push imm32; movabs rax, dispatcher; call rax; add rsp, 8; ret
    slot[0] = 0x68;
    slot[1..5].copy_from_slice(&index.to_le_bytes());
    slot[5] = 0x48;
    slot[6] = 0xb8;
    slot[7..15].copy_from_slice(&dispatcher.to_le_bytes());
    slot[15] = 0xff;
    slot[16] = 0xd0;
    slot[17..21].copy_from_slice(&[0x48, 0x83, 0xc4, 0x08]);
    slot[21] = 0xc3;
    Ok(())
}

#[derive(Debug, Default)]
struct ProbeDispatchState {
    nids: Vec<String>,
    hits: usize,
    maximum_hits: usize,
    next_direct_offset: u64,
    next_virtual_hint: u64,
    mapped_allocations: Vec<(u64, u64)>,
}

static PROBE_DISPATCH: OnceLock<RwLock<ProbeDispatchState>> = OnceLock::new();

#[cfg(target_arch = "x86_64")]
fn configure_probe_dispatch(stubs: &[ImportStub], maximum_hits: usize) {
    let state = PROBE_DISPATCH.get_or_init(|| RwLock::new(ProbeDispatchState::default()));
    let mut state = state
        .write()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    state.nids = stubs.iter().map(|stub| stub.nid.clone()).collect();
    state.hits = 0;
    state.maximum_hits = maximum_hits;
    state.next_direct_offset = 0;
    state.next_virtual_hint = 0x20_0000_0000;
    state.mapped_allocations.clear();
}

#[allow(clippy::too_many_arguments)]
extern "C" fn import_dispatch(
    argument0: u64,
    argument1: u64,
    argument2: u64,
    argument3: u64,
    argument4: u64,
    argument5: u64,
    index: u64,
) -> u64 {
    let state = PROBE_DISPATCH.get_or_init(|| RwLock::new(ProbeDispatchState::default()));
    let mut state = state
        .write()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let Some(nid) = usize::try_from(index)
        .ok()
        .and_then(|index| state.nids.get(index))
        .cloned()
    else {
        eprintln!("[REMu][IMPORT] invalid stub index {index}");
        terminate_probe(87);
    };
    state.hits += 1;
    eprintln!(
        "[REMu][IMPORT] hit={} nid={} args={argument0:#x},{argument1:#x},{argument2:#x},{argument3:#x},{argument4:#x},{argument5:#x}",
        state.hits, nid
    );
    if state.maximum_hits != 0 && state.hits >= state.maximum_hits {
        eprintln!("[REMu][PROBE] import hit limit reached");
        terminate_probe(86);
    }
    dispatch_probe_import(
        &nid,
        [
            argument0, argument1, argument2, argument3, argument4, argument5,
        ],
        &mut state,
    )
}

fn dispatch_probe_import(nid: &str, arguments: [u64; 6], state: &mut ProbeDispatchState) -> u64 {
    match nid {
        // sceKernelGetDirectMemorySize
        "pO96TwzOm5E" => 16 * 1024 * 1024 * 1024,
        // sceKernelAllocateDirectMemory
        "rTXw65xmLIA" => allocate_direct_memory(arguments, state),
        // sceKernelMapDirectMemory
        "L-Q3LEjIbgA" => map_direct_memory(arguments, state),
        // libc memset
        "8zTFvBIAIN8" => guest_memset(arguments[0], arguments[1], arguments[2]),
        // libc strlen
        "j4ViWNHEgww" => guest_strlen(arguments[0]),
        // libc fputs
        "QrZZdJ8XsX0" => guest_fputs(arguments[0]),
        // libKernel exit
        "uMei1W9uyNo" => {
            let bytes = arguments[0].to_le_bytes();
            let status = i32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]);
            eprintln!("[REMu][PROBE] guest requested exit({status})");
            terminate_probe(status);
        }
        _ => 0,
    }
}

fn allocate_direct_memory(arguments: [u64; 6], state: &mut ProbeDispatchState) -> u64 {
    let length = arguments[2];
    let alignment = arguments[3].max(0x4000);
    let output = arguments[5];
    if length == 0 || output == 0 || !alignment.is_power_of_two() {
        return u64::MAX;
    }
    let Some(offset) = align_up_u64(state.next_direct_offset, alignment) else {
        return u64::MAX;
    };
    let Some(next) = offset.checked_add(length) else {
        return u64::MAX;
    };
    if next > 16 * 1024 * 1024 * 1024 {
        return u64::MAX;
    }
    // SAFETY: The guest supplied an ABI output pointer in its live stack/image mapping.
    unsafe {
        std::ptr::write_unaligned(output as *mut u64, offset);
    }
    state.next_direct_offset = next;
    0
}

#[cfg(unix)]
fn map_direct_memory(arguments: [u64; 6], state: &mut ProbeDispatchState) -> u64 {
    let output = arguments[0];
    let length = arguments[1];
    let alignment = arguments[5].max(0x4000);
    if output == 0 || length == 0 || !alignment.is_power_of_two() {
        return u64::MAX;
    }
    let Ok(length_usize) = usize::try_from(length) else {
        return u64::MAX;
    };
    let Ok(alignment_usize) = usize::try_from(alignment) else {
        return u64::MAX;
    };
    let Some(reservation_size) = length_usize.checked_add(alignment_usize) else {
        return u64::MAX;
    };
    let hint = align_up_u64(state.next_virtual_hint, alignment).unwrap_or(0);
    // SAFETY: Anonymous mmap creates a new process-local reservation. The hint is non-destructive
    // because MAP_FIXED is deliberately absent.
    let reservation = unsafe {
        libc::mmap(
            hint as *mut std::ffi::c_void,
            reservation_size,
            libc::PROT_READ | libc::PROT_WRITE,
            libc::MAP_PRIVATE | libc::MAP_ANON,
            -1,
            0,
        )
    };
    if reservation == libc::MAP_FAILED {
        return u64::MAX;
    }
    let reservation_address = reservation as usize;
    let Some(aligned_address) = reservation_address
        .checked_add(alignment_usize - 1)
        .map(|value| value & !(alignment_usize - 1))
    else {
        // SAFETY: Releases the reservation returned by mmap above.
        unsafe {
            libc::munmap(reservation, reservation_size);
        }
        return u64::MAX;
    };
    let prefix = aligned_address - reservation_address;
    let suffix = reservation_size - prefix - length_usize;
    // SAFETY: Prefix/suffix are page-aligned subranges of the live anonymous reservation.
    unsafe {
        if prefix != 0 {
            libc::munmap(reservation, prefix);
        }
        if suffix != 0 {
            libc::munmap(
                (aligned_address + length_usize) as *mut std::ffi::c_void,
                suffix,
            );
        }
        std::ptr::write_unaligned(output as *mut u64, aligned_address as u64);
    }
    state.next_virtual_hint = (aligned_address as u64).saturating_add(length);
    state
        .mapped_allocations
        .push((aligned_address as u64, length));
    0
}

#[cfg(not(unix))]
fn map_direct_memory(_arguments: [u64; 6], _state: &mut ProbeDispatchState) -> u64 {
    u64::MAX
}

fn guest_memset(destination: u64, value: u64, length: u64) -> u64 {
    if destination == 0 || length == 0 {
        return destination;
    }
    let Ok(length) = usize::try_from(length) else {
        return 0;
    };
    // SAFETY: The call originates in guest code and supplies a live destination mapping.
    unsafe {
        std::ptr::write_bytes(destination as *mut u8, value.to_le_bytes()[0], length);
    }
    destination
}

fn guest_strlen(address: u64) -> u64 {
    if address == 0 {
        return 0;
    }
    // SAFETY: The guest libc call contract supplies a valid NUL-terminated string.
    unsafe { libc::strlen(address as *const libc::c_char) as u64 }
}

fn guest_fputs(address: u64) -> u64 {
    if address == 0 {
        return u64::MAX;
    }
    // SAFETY: The guest libc call contract supplies a valid NUL-terminated string.
    let text = unsafe { std::ffi::CStr::from_ptr(address as *const libc::c_char) };
    eprint!("{}", text.to_string_lossy());
    0
}

const fn align_up_u64(value: u64, alignment: u64) -> Option<u64> {
    let mask = alignment - 1;
    match value.checked_add(mask) {
        Some(value) => Some(value & !mask),
        None => None,
    }
}

#[cfg(target_arch = "x86_64")]
fn probe_entry(
    image: &LinkedImage,
    process_image_name: &str,
    maximum_import_hits: usize,
) -> Result<ProbeOutcome, LinkError> {
    #[repr(C)]
    struct EntryParameters {
        argument_count: u32,
        reserved: u32,
        argument0: u64,
        argument1: u64,
        argument2: u64,
    }
    type GuestEntry = unsafe extern "C" fn(*const EntryParameters, extern "C" fn(i32)) -> i32;

    let image_name =
        CString::new(process_image_name).map_err(|_| LinkError::InvalidProcessImageName)?;
    let parameters = EntryParameters {
        argument_count: 1,
        reserved: 0,
        argument0: image_name.as_ptr() as usize as u64,
        argument1: 0,
        argument2: 0,
    };
    configure_probe_dispatch(&image.import_stubs, maximum_import_hits);
    // SAFETY: Alarm only bounds this diagnostic process; the default SIGALRM action terminates it.
    unsafe {
        libc::alarm(10);
    }
    // SAFETY: The linker validated and mapped the ELF entry in executable guest memory. The process
    // entry ABI is the same System V x86-64 convention used by this Rosetta process.
    let entry: GuestEntry = unsafe { std::mem::transmute(image.memory.entry_address().get()) };
    // SAFETY: `parameters` and its image-name pointer stay live for the duration of the call.
    let return_value = unsafe { entry(&raw const parameters, guest_exit) };
    // SAFETY: Cancels the diagnostic alarm after a normal return.
    unsafe {
        libc::alarm(0);
    }
    let import_hits = PROBE_DISPATCH.get().map_or(0, |state| {
        state
            .read()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .hits
    });
    Ok(ProbeOutcome {
        return_value,
        import_hits,
    })
}

#[cfg(not(target_arch = "x86_64"))]
const fn probe_entry(
    _image: &LinkedImage,
    _process_image_name: &str,
    _maximum_import_hits: usize,
) -> Result<ProbeOutcome, LinkError> {
    Err(LinkError::ProbeRequiresX86_64)
}

#[cfg(target_arch = "x86_64")]
extern "C" fn guest_exit(code: i32) {
    eprintln!("[REMu][PROBE] guest exit handler called with code {code}");
    terminate_probe(code);
}

fn terminate_probe(code: i32) -> ! {
    #[cfg(unix)]
    // SAFETY: `_exit` terminates the isolated CLI probe without unwinding across guest frames.
    unsafe {
        libc::_exit(code);
    }

    #[cfg(not(unix))]
    std::process::exit(code)
}

#[cfg(all(test, target_os = "macos", target_arch = "x86_64"))]
mod tests {
    use std::io::Cursor;

    use remu_core::GuestAddress;
    use remu_loader::load_reader;

    use super::LinkedImage;

    #[test]
    fn links_relative_and_import_relocations_into_fixed_memory() {
        let mut bytes = vec![0_u8; 0x400];
        write_elf_header(&mut bytes);
        write_program_header(&mut bytes, 0, 1, 7, 0x100, 0x4000, 0x200, 0x300);
        write_program_header(&mut bytes, 1, 2, 6, 0x100, 0x4000, 0x70, 0x70);
        write_program_header(&mut bytes, 2, 7, 4, 0x260, 0x4160, 2, 0x20);
        bytes[0xe0..0xe8].copy_from_slice(&0x10_u64.to_le_bytes());
        write_dynamic(&mut bytes[0x100..]);
        bytes[0x180..0x18d].copy_from_slice(b"\0testNid#A#A\0");
        bytes[0x1d8..0x1dc].copy_from_slice(&1_u32.to_le_bytes());
        bytes[0x1dc] = 0x12;
        write_relocation(&mut bytes[0x1f0..], 0, 0x4180, 8, 0x1234);
        write_relocation(&mut bytes[0x1f0..], 1, 0x4188, (1_u64 << 32) | 7, 0);
        bytes[0x240..0x249].copy_from_slice(&[0x64, 0x48, 0x8b, 0x04, 0x25, 0, 0, 0, 0]);
        bytes[0x260..0x262].copy_from_slice(&[0xaa, 0xbb]);
        let executable = load_reader(&mut Cursor::new(bytes)).unwrap();

        let linked = LinkedImage::link_main(&executable).unwrap();
        assert_eq!(linked.relocation_stats.relative, 1);
        assert_eq!(linked.relocation_stats.jump_slots, 1);
        assert_eq!(linked.import_stubs.len(), 1);
        assert_eq!(linked.tls.static_block_size, 0x20);
        assert_eq!(linked.tls.patched_loads, 1);
        let base = linked.memory.image_base().get();
        let mut relative = [0_u8; 8];
        linked
            .memory
            .read(GuestAddress::new(base + 0x4180), &mut relative)
            .unwrap();
        assert_eq!(u64::from_le_bytes(relative), base + 0x1234);
        let mut import = [0_u8; 8];
        linked
            .memory
            .read(GuestAddress::new(base + 0x4188), &mut import)
            .unwrap();
        assert_eq!(
            u64::from_le_bytes(import),
            linked.import_stubs[0].address.get()
        );
        let mut patched_tls_load = [0_u8; 9];
        linked
            .memory
            .read(GuestAddress::new(base + 0x4140), &mut patched_tls_load)
            .unwrap();
        assert_eq!(patched_tls_load[0], 0xe8);
        assert_eq!(&patched_tls_load[5..], &[0x90; 4]);
        let mut tls_template = [0_u8; 2];
        linked
            .memory
            .read(
                GuestAddress::new(linked.tls.thread_pointer.get() - 0x20),
                &mut tls_template,
            )
            .unwrap();
        assert_eq!(tls_template, [0xaa, 0xbb]);
    }

    fn write_elf_header(bytes: &mut [u8]) {
        bytes[..4].copy_from_slice(&[0x7f, b'E', b'L', b'F']);
        bytes[4] = 2;
        bytes[5] = 1;
        bytes[0x12..0x14].copy_from_slice(&62_u16.to_le_bytes());
        bytes[0x18..0x20].copy_from_slice(&0x4000_u64.to_le_bytes());
        bytes[0x20..0x28].copy_from_slice(&64_u64.to_le_bytes());
        bytes[0x34..0x36].copy_from_slice(&64_u16.to_le_bytes());
        bytes[0x36..0x38].copy_from_slice(&56_u16.to_le_bytes());
        bytes[0x38..0x3a].copy_from_slice(&3_u16.to_le_bytes());
    }

    #[allow(clippy::too_many_arguments)]
    fn write_program_header(
        bytes: &mut [u8],
        index: usize,
        program_type: u32,
        flags: u32,
        file_offset: u64,
        virtual_address: u64,
        file_size: u64,
        memory_size: u64,
    ) {
        let offset = 64 + index * 56;
        bytes[offset..offset + 4].copy_from_slice(&program_type.to_le_bytes());
        bytes[offset + 4..offset + 8].copy_from_slice(&flags.to_le_bytes());
        bytes[offset + 8..offset + 16].copy_from_slice(&file_offset.to_le_bytes());
        bytes[offset + 16..offset + 24].copy_from_slice(&virtual_address.to_le_bytes());
        bytes[offset + 32..offset + 40].copy_from_slice(&file_size.to_le_bytes());
        bytes[offset + 40..offset + 48].copy_from_slice(&memory_size.to_le_bytes());
        bytes[offset + 48..offset + 56].copy_from_slice(&0x4000_u64.to_le_bytes());
    }

    fn write_dynamic(bytes: &mut [u8]) {
        let entries = [
            (5_i64, 0x4080_u64),
            (10, 0x40),
            (6, 0x40c0),
            (0x6100_003f, 0x30),
            (7, 0x40f0),
            (8, 0x30),
            (0, 0),
        ];
        for (index, (tag, value)) in entries.into_iter().enumerate() {
            let offset = index * 16;
            bytes[offset..offset + 8].copy_from_slice(&tag.to_le_bytes());
            bytes[offset + 8..offset + 16].copy_from_slice(&value.to_le_bytes());
        }
    }

    fn write_relocation(bytes: &mut [u8], index: usize, offset: u64, info: u64, addend: i64) {
        let entry = index * 24;
        bytes[entry..entry + 8].copy_from_slice(&offset.to_le_bytes());
        bytes[entry + 8..entry + 16].copy_from_slice(&info.to_le_bytes());
        bytes[entry + 16..entry + 24].copy_from_slice(&addend.to_le_bytes());
    }
}
