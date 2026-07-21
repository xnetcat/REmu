// Copyright (C) 2026 REmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

use std::cell::Cell;
use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::ffi::CString;
use std::path::{Path, PathBuf};
use std::sync::{Condvar, Mutex, OnceLock, RwLock};
use std::thread::ThreadId;
use std::time::Duration;

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
const TLS_HELPER_SIZE: u64 = 96;
const TLS_HELPER_SIZE_USIZE: usize = 96;
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
        game_root: &Path,
        maximum_import_hits: usize,
    ) -> Result<ProbeOutcome, LinkError> {
        probe_entry(self, process_image_name, game_root, maximum_import_hits)
    }
}

fn install_tls_support(
    memory: &mut GuestImage,
    executable: &LoadedExecutable,
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
    seen_nids: BTreeSet<String>,
    import_counts: BTreeMap<String, usize>,
    hits: usize,
    maximum_hits: usize,
    next_direct_offset: u64,
    next_virtual_hint: u64,
    mapped_allocations: Vec<(u64, u64)>,
    next_thread_handle: u64,
    thread_handles: BTreeMap<u64, std::thread::JoinHandle<u64>>,
    agc_register_defaults: u64,
    ctype_table: u64,
    tls_template_address: u64,
    tls_static_block_size: u64,
    video_out: VideoOutState,
    game_root: PathBuf,
    next_np_handle: i32,
}

#[derive(Debug, Default)]
struct VideoOutState {
    pixel_format: u64,
    tiling_mode: u32,
    width: u32,
    height: u32,
    display_buffers: Vec<u64>,
}

static PROBE_DISPATCH: OnceLock<RwLock<ProbeDispatchState>> = OnceLock::new();

#[derive(Debug, Default)]
struct GuestSyncState {
    mutexes: HashMap<u64, GuestMutexState>,
    conditions: HashMap<u64, u64>,
    semaphores: HashMap<u32, GuestSemaphoreState>,
    next_semaphore_handle: u32,
}

#[derive(Debug, Default)]
struct GuestMutexState {
    owner: Option<ThreadId>,
    recursion: u32,
}

#[derive(Debug)]
struct GuestSemaphoreState {
    count: i32,
    maximum: i32,
}

static GUEST_SYNC: OnceLock<(Mutex<GuestSyncState>, Condvar)> = OnceLock::new();

thread_local! {
    static CURRENT_GUEST_THREAD_POINTER: Cell<u64> = const { Cell::new(0) };
}

extern "C" fn current_guest_thread_pointer() -> u64 {
    CURRENT_GUEST_THREAD_POINTER.get()
}

fn set_current_guest_thread_pointer(thread_pointer: u64) {
    CURRENT_GUEST_THREAD_POINTER.set(thread_pointer);
}

#[cfg(target_arch = "x86_64")]
fn configure_probe_dispatch(image: &LinkedImage, game_root: &Path, maximum_hits: usize) {
    let state = PROBE_DISPATCH.get_or_init(|| RwLock::new(ProbeDispatchState::default()));
    let mut state = state
        .write()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    state.nids = image
        .import_stubs
        .iter()
        .map(|stub| stub.nid.clone())
        .collect();
    state.seen_nids.clear();
    state.import_counts.clear();
    state.hits = 0;
    state.maximum_hits = maximum_hits;
    state.next_direct_offset = 0;
    state.next_virtual_hint = 0x20_0000_0000;
    state.mapped_allocations.clear();
    state.next_thread_handle = 1;
    state.thread_handles.clear();
    state.agc_register_defaults = 0;
    state.ctype_table = 0;
    state.tls_template_address = image.tls.thread_pointer.get() - GUEST_TLS_PREFIX_SIZE;
    state.tls_static_block_size = image.tls.static_block_size;
    state.video_out = VideoOutState::default();
    game_root.clone_into(&mut state.game_root);
    state.next_np_handle = 1;
    let sync = GUEST_SYNC.get_or_init(|| (Mutex::new(GuestSyncState::default()), Condvar::new()));
    *sync
        .0
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner) = GuestSyncState::default();
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
    let call_args = [
        argument0, argument1, argument2, argument3, argument4, argument5,
    ];
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
    *state.import_counts.entry(nid.clone()).or_default() += 1;
    if state.seen_nids.insert(nid.clone()) {
        eprintln!(
            "[REMu][IMPORT] first_hit={} nid={} args={argument0:#x},{argument1:#x},{argument2:#x},{argument3:#x},{argument4:#x},{argument5:#x}",
            state.hits, nid
        );
    }
    if state.maximum_hits != 0 && state.hits >= state.maximum_hits {
        eprintln!("[REMu][PROBE] import hit limit reached at nid={nid}");
        let mut counts = state
            .import_counts
            .iter()
            .map(|(nid, count)| (nid, *count))
            .collect::<Vec<_>>();
        counts.sort_unstable_by_key(|(_, count)| std::cmp::Reverse(*count));
        for (nid, count) in counts.into_iter().take(8) {
            eprintln!("[REMu][PROBE] hot_import nid={nid} hits={count}");
        }
        terminate_probe(86);
    }
    if nid == "6UgtwV+0zb4" {
        let handle = state.next_thread_handle;
        state.next_thread_handle = state.next_thread_handle.saturating_add(1);
        if call_args[0] != 0 {
            write_guest_u64(call_args[0], handle);
        }
        let entry = call_args[2];
        let thread_argument = call_args[3];
        let guest_tls = clone_guest_tls(state.tls_template_address, state.tls_static_block_size);
        let thread = std::thread::spawn(move || {
            set_current_guest_thread_pointer(guest_tls.thread_pointer);
            let result = run_guest_thread(entry, thread_argument);
            drop(guest_tls);
            result
        });
        state.thread_handles.insert(handle, thread);
        return 0;
    }
    if nid == "onNY9Byn-W8" {
        let handle = state.thread_handles.remove(&call_args[0]);
        drop(state);
        let return_value = handle.map_or(0, |thread| thread.join().unwrap_or(0));
        if call_args[1] != 0 {
            write_guest_u64(call_args[1], return_value);
        }
        return 0;
    }
    if is_guest_sync_import(&nid) {
        drop(state);
        return dispatch_guest_sync_import(&nid, call_args);
    }
    // A real AudioOut submission blocks for approximately one 256-frame buffer. Without pacing,
    // the emulated audio worker monopolizes the import boundary and starves the render worker.
    if nid == "QOQtbeDqsT4" {
        drop(state);
        std::thread::sleep(Duration::from_micros(5_333));
        return 0;
    }
    dispatch_probe_import(&nid, call_args, &mut state)
}

struct GuestThreadTls {
    _storage: Vec<u8>,
    thread_pointer: u64,
}

fn clone_guest_tls(template_address: u64, static_block_size: u64) -> GuestThreadTls {
    let size = usize::try_from(GUEST_TLS_PREFIX_SIZE + GUEST_TLS_TCB_SIZE)
        .expect("guest TLS allocation size fits usize");
    let mut storage = vec![0_u8; size];
    // SAFETY: The process-wide TLS template remains mapped while every guest worker is alive.
    unsafe {
        std::ptr::copy_nonoverlapping(
            template_address as *const u8,
            storage.as_mut_ptr(),
            storage.len(),
        );
    }
    let thread_pointer = storage.as_mut_ptr() as usize as u64 + GUEST_TLS_PREFIX_SIZE;
    write_guest_u64(thread_pointer, thread_pointer);
    write_guest_u64(thread_pointer + 0x08, thread_pointer + 0x100);
    write_guest_u64(thread_pointer + 0x10, thread_pointer);
    write_guest_u64(thread_pointer + 0x60, thread_pointer);
    write_guest_u64(thread_pointer + 0x100, 1);
    write_guest_u64(
        thread_pointer + 0x108,
        usize::from(static_block_size != 0) as u64,
    );
    if static_block_size != 0 {
        write_guest_u64(thread_pointer + 0x110, thread_pointer - static_block_size);
    }
    GuestThreadTls {
        _storage: storage,
        thread_pointer,
    }
}

fn is_guest_sync_import(nid: &str) -> bool {
    matches!(
        nid,
        "cmo1RIYva9o"
            | "2Of0f+3mhhE"
            | "9UK1vLZQft4"
            | "upoVrzMHFeE"
            | "tn3VlD0hG60"
            | "ttHNfU+qDBU"
            | "ltCfaGr2JGE"
            | "7H0iTOciTLo"
            | "K-jXhbt2gn4"
            | "2Z+PpY6CaJg"
            | "2Tb92quprl0"
            | "0TyVk4MSLt0"
            | "g+PZd2hiacg"
            | "RXXqi4CtF8w"
            | "WKAXJ4XBPQ4"
            | "BmMjYxmew1w"
            | "kDh-NfxgMtE"
            | "JGgj7Uvrl+A"
            | "188x57JYp0g"
            | "Zxa0VhQVTsk"
            | "4czppHBiriw"
    )
}

fn dispatch_guest_sync_import(nid: &str, arguments: [u64; 6]) -> u64 {
    match nid {
        "cmo1RIYva9o" | "ttHNfU+qDBU" => guest_mutex_init(arguments[0]),
        "2Of0f+3mhhE" | "ltCfaGr2JGE" => guest_mutex_destroy(arguments[0]),
        "9UK1vLZQft4" | "7H0iTOciTLo" => guest_mutex_lock(arguments[0], false),
        "upoVrzMHFeE" | "K-jXhbt2gn4" => guest_mutex_lock(arguments[0], true),
        "tn3VlD0hG60" | "2Z+PpY6CaJg" => guest_mutex_unlock(arguments[0]),
        "2Tb92quprl0" | "0TyVk4MSLt0" => guest_condition_init(arguments[0]),
        "g+PZd2hiacg" | "RXXqi4CtF8w" => guest_condition_destroy(arguments[0]),
        "WKAXJ4XBPQ4" => guest_condition_wait(arguments[0], arguments[1], None),
        "BmMjYxmew1w" => guest_condition_wait(
            arguments[0],
            arguments[1],
            Some(u64::from(low_u32(arguments[2]))),
        ),
        "kDh-NfxgMtE" | "JGgj7Uvrl+A" => guest_condition_signal(arguments[0]),
        "188x57JYp0g" => guest_semaphore_create(arguments),
        "Zxa0VhQVTsk" => guest_semaphore_wait(arguments),
        "4czppHBiriw" => guest_semaphore_signal(arguments),
        _ => 0,
    }
}

fn guest_sync() -> &'static (Mutex<GuestSyncState>, Condvar) {
    GUEST_SYNC.get_or_init(|| (Mutex::new(GuestSyncState::default()), Condvar::new()))
}

fn guest_mutex_init(address: u64) -> u64 {
    if address == 0 {
        return u64::MAX;
    }
    let sync = guest_sync();
    sync.0
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .mutexes
        .entry(address)
        .or_default();
    0
}

fn guest_mutex_destroy(address: u64) -> u64 {
    let sync = guest_sync();
    sync.0
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .mutexes
        .remove(&address);
    0
}

fn guest_mutex_lock(address: u64, try_only: bool) -> u64 {
    if address == 0 {
        return u64::MAX;
    }
    let sync = guest_sync();
    let current = std::thread::current().id();
    let mut state = sync
        .0
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    loop {
        let mutex = state.mutexes.entry(address).or_default();
        match mutex.owner.as_ref() {
            None => {
                mutex.owner = Some(current);
                mutex.recursion = 1;
                return 0;
            }
            Some(owner) if *owner == current => {
                mutex.recursion = mutex.recursion.saturating_add(1);
                return 0;
            }
            Some(_) if try_only => return u64::from(libc::EBUSY as u32),
            Some(_) => {
                state = sync
                    .1
                    .wait(state)
                    .unwrap_or_else(std::sync::PoisonError::into_inner);
            }
        }
    }
}

fn guest_mutex_unlock(address: u64) -> u64 {
    let sync = guest_sync();
    let current = std::thread::current().id();
    let mut state = sync
        .0
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    if let Some(mutex) = state.mutexes.get_mut(&address)
        && mutex.owner.as_ref().is_some_and(|owner| *owner == current)
    {
        if mutex.recursion > 1 {
            mutex.recursion -= 1;
        } else {
            mutex.owner = None;
            mutex.recursion = 0;
            sync.1.notify_all();
        }
    }
    0
}

fn guest_condition_init(address: u64) -> u64 {
    let sync = guest_sync();
    sync.0
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .conditions
        .entry(address)
        .or_default();
    0
}

fn guest_condition_destroy(address: u64) -> u64 {
    let sync = guest_sync();
    sync.0
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .conditions
        .remove(&address);
    0
}

fn guest_condition_wait(condition: u64, mutex_address: u64, timeout_usec: Option<u64>) -> u64 {
    let sync = guest_sync();
    let current = std::thread::current().id();
    let mut state = sync
        .0
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let generation = *state.conditions.entry(condition).or_default();
    if let Some(mutex) = state.mutexes.get_mut(&mutex_address)
        && mutex.owner.as_ref().is_some_and(|owner| *owner == current)
    {
        mutex.owner = None;
        mutex.recursion = 0;
    }
    sync.1.notify_all();

    if let Some(timeout) = timeout_usec {
        let duration = Duration::from_micros(timeout);
        let (next_state, _) = sync
            .1
            .wait_timeout_while(state, duration, |state| {
                state.conditions.get(&condition).copied().unwrap_or(0) == generation
            })
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        state = next_state;
    } else {
        state = sync
            .1
            .wait_while(state, |state| {
                state.conditions.get(&condition).copied().unwrap_or(0) == generation
            })
            .unwrap_or_else(std::sync::PoisonError::into_inner);
    }

    loop {
        let mutex = state.mutexes.entry(mutex_address).or_default();
        if mutex.owner.is_none() {
            mutex.owner = Some(current);
            mutex.recursion = 1;
            return 0;
        }
        state = sync
            .1
            .wait(state)
            .unwrap_or_else(std::sync::PoisonError::into_inner);
    }
}

fn guest_condition_signal(condition: u64) -> u64 {
    let sync = guest_sync();
    let mut state = sync
        .0
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let generation = state.conditions.entry(condition).or_default();
    *generation = generation.wrapping_add(1);
    sync.1.notify_all();
    0
}

fn guest_semaphore_create(arguments: [u64; 6]) -> u64 {
    let output = arguments[0];
    let initial = low_i32(arguments[3]);
    let maximum = low_i32(arguments[4]);
    if output == 0 || initial < 0 || maximum <= 0 || initial > maximum {
        return u64::MAX;
    }
    let sync = guest_sync();
    let mut state = sync
        .0
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    state.next_semaphore_handle = state.next_semaphore_handle.wrapping_add(1).max(1);
    let handle = state.next_semaphore_handle;
    state.semaphores.insert(
        handle,
        GuestSemaphoreState {
            count: initial,
            maximum,
        },
    );
    // SAFETY: sceKernelCreateSema supplies a writable 32-bit handle output.
    unsafe {
        std::ptr::write_unaligned(output as *mut u32, handle);
    }
    0
}

fn guest_semaphore_wait(arguments: [u64; 6]) -> u64 {
    let handle = low_u32(arguments[0]);
    let needed = low_i32(arguments[1]);
    if needed <= 0 {
        return u64::MAX;
    }
    let sync = guest_sync();
    let mut state = sync
        .0
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    loop {
        let Some(semaphore) = state.semaphores.get_mut(&handle) else {
            return u64::MAX;
        };
        if semaphore.count >= needed {
            semaphore.count -= needed;
            return 0;
        }
        state = sync
            .1
            .wait(state)
            .unwrap_or_else(std::sync::PoisonError::into_inner);
    }
}

fn guest_semaphore_signal(arguments: [u64; 6]) -> u64 {
    let handle = low_u32(arguments[0]);
    let amount = low_i32(arguments[1]);
    let sync = guest_sync();
    let mut state = sync
        .0
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let Some(semaphore) = state.semaphores.get_mut(&handle) else {
        return u64::MAX;
    };
    semaphore.count = semaphore
        .count
        .saturating_add(amount)
        .min(semaphore.maximum);
    sync.1.notify_all();
    0
}

fn low_u32(value: u64) -> u32 {
    let bytes = value.to_le_bytes();
    u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]])
}

fn low_i32(value: u64) -> i32 {
    let bytes = value.to_le_bytes();
    i32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]])
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
        // libc memcpy
        "Q3VBxCXhUHs" => guest_memcpy(arguments[0], arguments[1], arguments[2]),
        // libc memmove
        "+P6FRGH4LfA" => guest_memmove(arguments[0], arguments[1], arguments[2]),
        // The native x86-64 process and guest share the libc allocation ABI.
        "gQX+4GDQjpM" => guest_malloc(arguments[0]),
        "tIhsqj0qsFE" => guest_free(arguments[0]),
        "2X5agFjKxMc" => guest_calloc(arguments[0], arguments[1]),
        "Y7aJ1uydPMo" => guest_realloc(arguments[0], arguments[1]),
        "Ujf3KzMvRmI" | "2Btkg8k24Zg" => guest_aligned_alloc(arguments[0], arguments[1]),
        "cVSk9y8URbc" => guest_posix_memalign(arguments[0], arguments[1], arguments[2]),
        // libc strlen
        "j4ViWNHEgww" => guest_strlen(arguments[0]),
        "Ovb2dSJOAuE" => guest_strcmp(arguments[0], arguments[1]),
        "aesyjrHVWy4" => guest_strncmp(arguments[0], arguments[1], arguments[2]),
        "kiZSXIWd9vg" => guest_strcpy(arguments[0], arguments[1]),
        "6sJWiWSRuqk" => guest_strncpy(arguments[0], arguments[1], arguments[2]),
        "Ls4tzzhimqQ" => guest_strcat(arguments[0], arguments[1]),
        "AV6ipCNa4Rw" => guest_strcasecmp(arguments[0], arguments[1]),
        // libc stdio uses host FILE objects after resolving console paths inside the game dump.
        "xeYO4u7uyJ0" => guest_fopen(arguments[0], arguments[1], state),
        "rQFVBXp-Cxg" => guest_fseek(arguments[0], arguments[1], arguments[2]),
        "Qazy8LmXTvw" => guest_ftell(arguments[0]),
        "uodLYyUip20" => guest_fclose(arguments[0]),
        "lbB+UlZqVG0" => guest_fread(arguments[0], arguments[1], arguments[2], arguments[3]),
        // libc fputs
        "QrZZdJ8XsX0" => guest_fputs(arguments[0]),
        // Itanium C++ one-time initialization guards.
        "3GPpjQdAMTw" => guest_guard_acquire(arguments[0]),
        "9rAeANT2tyE" => guest_guard_release(arguments[0]),
        "2emaaluWzUw" => guest_guard_abort(arguments[0]),
        // The single-thread bootstrap uses these attributes only to configure the first worker.
        "F8bUHwAG284" | "nsYoNRywwNg" => initialize_synthetic_handle(arguments[0]),
        "smWEktiyyG0" | "62KCwEMmzcM" => destroy_synthetic_handle(arguments[0]),
        // scePthreadAttrGetstack returns both values through caller-owned outputs.
        "-quPa4SEJUw" => pthread_attr_get_stack(arguments),
        // Initial single-device handles for input, video, and audio.
        "xk0AcarP3V4" | "Up36PTk687E" | "ekNvsT22rsY" => 1,
        // AGC shader construction and VideoOut mutate caller-owned descriptors. Games consume the
        // resulting pointers and command data immediately, so successful no-ops are insufficient.
        "f3dg2CSgRKY" => agc_create_shader(arguments),
        "PjS5uASwcV8" => video_out_set_buffer_attribute2(arguments, state),
        "rKBUtgRrtbk" => video_out_register_buffers2(arguments, state),
        // Initial user, NP context, and trophy context IDs.
        "CdWp0oHWGr0" | "5zBnau1uIEo" | "Bagshr7OQ6Q" => write_output_i32(arguments[0], 1),
        // sceKernelCreateEqueue
        "D0OdFMjp46I" => write_output_u64(arguments[0], 1),
        // Minimal offline NP identities. These services only require stable nonzero IDs at boot.
        "hT0IAEvN+M0" => np_create_handle(arguments, state),
        // Minimal AGC register-default descriptor; zero groups select the game's fallback path.
        "2JtWUUiYBXs" | "wRbq6ZjNop4" => agc_register_defaults(state),
        // Inferred from the immediate two-byte character-class lookup at every call site.
        "1uJgoVq3bQU" => guest_ctype_table(state),
        // sceKernelUsleep
        "1jfXLRVzisc" => guest_usleep(arguments[0]),
        // libKernel exit
        "uMei1W9uyNo" => {
            let bytes = arguments[0].to_le_bytes();
            let status = i32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]);
            eprintln!("[REMu][PROBE] guest requested exit({status})");
            terminate_probe(status);
        }
        // Unimplemented calls are successful no-ops in the bounded discovery probe.
        _ => 0,
    }
}

fn agc_register_defaults(state: &mut ProbeDispatchState) -> u64 {
    if state.agc_register_defaults == 0 {
        state.agc_register_defaults = guest_calloc(1, 0x40);
    }
    state.agc_register_defaults
}

const AGC_SHADER_FILE_HEADER: u32 = 0x3433_3231;
const AGC_SHADER_VERSION: u32 = 0x18;
const AGC_SHADER_USER_DATA_OFFSET: u64 = 0x08;
const AGC_SHADER_CODE_OFFSET: u64 = 0x10;
const AGC_SHADER_CX_REGISTERS_OFFSET: u64 = 0x18;
const AGC_SHADER_SH_REGISTERS_OFFSET: u64 = 0x20;
const AGC_SHADER_SPECIALS_OFFSET: u64 = 0x28;
const AGC_SHADER_INPUT_SEMANTICS_OFFSET: u64 = 0x30;
const AGC_SHADER_OUTPUT_SEMANTICS_OFFSET: u64 = 0x38;
const AGC_SHADER_TYPE_OFFSET: u64 = 0x5a;
const AGC_SHADER_SH_REGISTER_COUNT_OFFSET: u64 = 0x5c;

fn agc_create_shader(arguments: [u64; 6]) -> u64 {
    let destination = arguments[0];
    let header = arguments[1];
    let code = arguments[2];
    if header == 0 || code == 0 {
        return u64::MAX;
    }
    if read_guest_u32(header) != AGC_SHADER_FILE_HEADER
        || read_guest_u32(header + 4) != AGC_SHADER_VERSION
    {
        eprintln!(
            "[REMu][AGC] rejected shader header at {header:#x}: magic={:#x} version={:#x}",
            read_guest_u32(header),
            read_guest_u32(header + 4)
        );
        return u64::MAX;
    }

    for offset in [
        AGC_SHADER_CX_REGISTERS_OFFSET,
        AGC_SHADER_SH_REGISTERS_OFFSET,
        AGC_SHADER_USER_DATA_OFFSET,
        AGC_SHADER_SPECIALS_OFFSET,
        AGC_SHADER_INPUT_SEMANTICS_OFFSET,
        AGC_SHADER_OUTPUT_SEMANTICS_OFFSET,
    ] {
        relocate_guest_pointer(header + offset);
    }
    write_guest_u64(header + AGC_SHADER_CODE_OFFSET, code);

    let user_data = read_guest_u64(header + AGC_SHADER_USER_DATA_OFFSET);
    if user_data != 0 {
        for offset in [0, 8, 0x10, 0x18, 0x20] {
            relocate_guest_pointer(user_data + offset);
        }
    }

    if !patch_agc_shader_program_registers(header, code) {
        return u64::MAX;
    }
    if destination != 0 {
        write_guest_u64(destination, header);
    }
    0
}

fn relocate_guest_pointer(field: u64) {
    let relative = read_guest_u64(field);
    if relative != 0 {
        write_guest_u64(field, field.wrapping_add(relative));
    }
}

fn patch_agc_shader_program_registers(header: u64, code: u64) -> bool {
    let registers = read_guest_u64(header + AGC_SHADER_SH_REGISTERS_OFFSET);
    let shader_type = read_guest_u8(header + AGC_SHADER_TYPE_OFFSET);
    let register_count = read_guest_u8(header + AGC_SHADER_SH_REGISTER_COUNT_OFFSET);
    if registers == 0 || register_count < 2 {
        return false;
    }
    let expected = match shader_type {
        0 => (0x20c, 0x20d),
        1 => (0x008, 0x009),
        2 | 6 => (0x0c8, 0x0c9),
        4 => (0x08a, 0x08b),
        7 => (0x148, 0x149),
        _ => return false,
    };
    if (read_guest_u32(registers), read_guest_u32(registers + 8)) != expected {
        return false;
    }
    write_guest_u32(registers + 4, ((code >> 8) & 0xffff_ffff) as u32);
    write_guest_u32(registers + 12, ((code >> 40) & 0xff) as u32);
    true
}

fn video_out_set_buffer_attribute2(arguments: [u64; 6], state: &mut ProbeDispatchState) -> u64 {
    let attribute = arguments[0];
    if attribute == 0 {
        return u64::MAX;
    }
    guest_memset(attribute, 0, 0x50);
    write_guest_u32(attribute + 0x04, low_u32(arguments[2]));
    write_guest_u32(attribute + 0x0c, low_u32(arguments[3]));
    write_guest_u32(attribute + 0x10, low_u32(arguments[4]));
    write_guest_u64(attribute + 0x18, arguments[5]);
    write_guest_u64(attribute + 0x20, arguments[1]);

    state.video_out.pixel_format = arguments[1];
    state.video_out.tiling_mode = low_u32(arguments[2]);
    state.video_out.width = low_u32(arguments[3]);
    state.video_out.height = low_u32(arguments[4]);
    0
}

fn video_out_register_buffers2(arguments: [u64; 6], state: &mut ProbeDispatchState) -> u64 {
    let set_index = low_u32(arguments[1]);
    let buffer_index_start = usize::try_from(low_u32(arguments[2])).unwrap_or(usize::MAX);
    let entries = arguments[3];
    let buffer_count = usize::try_from(low_u32(arguments[4])).unwrap_or(usize::MAX);
    let attribute = arguments[5];
    if arguments[0] == 0
        || entries == 0
        || attribute == 0
        || buffer_index_start > 15
        || buffer_count == 0
        || buffer_count > 16
        || buffer_index_start.saturating_add(buffer_count) > 16
    {
        return u64::MAX;
    }

    state.video_out.pixel_format = read_guest_u64(attribute + 0x20);
    state.video_out.tiling_mode = read_guest_u32(attribute + 0x04);
    state.video_out.width = read_guest_u32(attribute + 0x0c);
    state.video_out.height = read_guest_u32(attribute + 0x10);
    state
        .video_out
        .display_buffers
        .resize(buffer_index_start + buffer_count, 0);
    for index in 0..buffer_count {
        let entry = entries + u64::try_from(index).unwrap_or(u64::MAX) * 0x20;
        state.video_out.display_buffers[buffer_index_start + index] = read_guest_u64(entry);
    }
    eprintln!(
        "[REMu][VIDEO] registered {} buffers at {}x{} format={:#x} tiling={} addresses={:x?}",
        buffer_count,
        state.video_out.width,
        state.video_out.height,
        state.video_out.pixel_format,
        state.video_out.tiling_mode,
        state.video_out.display_buffers
    );
    u64::from(set_index)
}

fn guest_ctype_table(state: &mut ProbeDispatchState) -> u64 {
    if state.ctype_table == 0 {
        // Sony's libc follows the Dinkumware mask layout and indexes this table over
        // [-128, 255]. Return the c == 0 entry, leaving 128 valid entries before it.
        let storage = guest_calloc(384, 2);
        if storage == 0 {
            return 0;
        }
        state.ctype_table = storage + 128 * 2;
        for character in 0_u16..=0x7f {
            write_guest_u16(
                state.ctype_table + u64::from(character) * 2,
                guest_ctype_flags(character),
            );
        }
    }
    state.ctype_table
}

fn guest_ctype_flags(character: u16) -> u16 {
    const XDIGIT: u16 = 0x001;
    const UPPER: u16 = 0x002;
    const SPACE: u16 = 0x004;
    const PUNCT: u16 = 0x008;
    const LOWER: u16 = 0x010;
    const DIGIT: u16 = 0x020;
    const CONTROL_SPACE: u16 = 0x040;
    const CONTROL: u16 = 0x080;
    const BLANK: u16 = 0x400;

    let upper = (u16::from(b'A')..=u16::from(b'Z')).contains(&character);
    let lower = (u16::from(b'a')..=u16::from(b'z')).contains(&character);
    let digit = (u16::from(b'0')..=u16::from(b'9')).contains(&character);
    let hexadecimal = digit
        || (u16::from(b'A')..=u16::from(b'F')).contains(&character)
        || (u16::from(b'a')..=u16::from(b'f')).contains(&character);
    let alphanumeric = upper || lower || digit;
    let control_space = (0x09..=0x0d).contains(&character);
    let control = character <= 0x08 || (0x0e..=0x1f).contains(&character) || character == 0x7f;
    let printable = (0x20..=0x7e).contains(&character);

    let mut flags = 0;
    if upper {
        flags |= UPPER;
    }
    if lower {
        flags |= LOWER;
    }
    if digit {
        flags |= DIGIT;
    }
    if hexadecimal {
        flags |= XDIGIT;
    }
    if character == u16::from(b' ') {
        flags |= SPACE | BLANK;
    }
    if character == u16::from(b'\t') {
        flags |= BLANK;
    }
    if control_space {
        flags |= CONTROL_SPACE;
    }
    if printable && !alphanumeric && character != u16::from(b' ') {
        flags |= PUNCT;
    }
    if control {
        flags |= CONTROL;
    }
    flags
}

fn run_guest_thread(entry: u64, argument: u64) -> u64 {
    type GuestThreadEntry = unsafe extern "C" fn(u64) -> u64;

    if entry == 0 {
        return 0;
    }
    // SAFETY: The guest supplied a mapped executable entry point through scePthreadCreate. Rosetta
    // executes it on a native host thread while the import boundary provides HLE synchronization.
    let entry: GuestThreadEntry = unsafe { std::mem::transmute(entry) };
    // SAFETY: The guest owns the argument and execution contract for its thread entry.
    unsafe { entry(argument) }
}

fn initialize_synthetic_handle(address: u64) -> u64 {
    if address == 0 {
        return u64::MAX;
    }
    write_guest_u64(address, address);
    0
}

fn destroy_synthetic_handle(address: u64) -> u64 {
    if address == 0 {
        return u64::MAX;
    }
    write_guest_u64(address, 0);
    0
}

fn pthread_attr_get_stack(arguments: [u64; 6]) -> u64 {
    if arguments[0] == 0 || arguments[1] == 0 || arguments[2] == 0 {
        return u64::MAX;
    }
    write_guest_u64(arguments[1], 0);
    write_guest_u64(arguments[2], 0x80000);
    0
}

fn np_create_handle(arguments: [u64; 6], state: &mut ProbeDispatchState) -> u64 {
    let output = if arguments[0] != 0 {
        arguments[0]
    } else {
        arguments[1]
    };
    if output == 0 {
        return u64::MAX;
    }
    let handle = state.next_np_handle;
    state.next_np_handle = state.next_np_handle.saturating_add(1);
    write_output_i32(output, handle)
}

fn guest_guard_acquire(address: u64) -> u64 {
    if address == 0 {
        return 0;
    }
    // SAFETY: C++ emits a writable eight-byte guard object and passes its live address.
    let value = unsafe { std::ptr::read_unaligned(address as *const u64) };
    if value & 1 != 0 || value & 0xff00 != 0 {
        return 0;
    }
    write_guest_u64(address, 0x100);
    1
}

fn guest_guard_release(address: u64) -> u64 {
    if address != 0 {
        write_guest_u64(address, 1);
    }
    0
}

fn guest_guard_abort(address: u64) -> u64 {
    if address != 0 {
        write_guest_u64(address, 0);
    }
    0
}

fn write_guest_u64(address: u64, value: u64) {
    // SAFETY: HLE calls use guest pointers into the process's live mapped address space.
    unsafe {
        std::ptr::write_unaligned(address as *mut u64, value);
    }
}

fn read_guest_u8(address: u64) -> u8 {
    // SAFETY: HLE calls use guest pointers into the process's live mapped address space.
    unsafe { std::ptr::read_unaligned(address as *const u8) }
}

fn read_guest_u32(address: u64) -> u32 {
    // SAFETY: HLE calls use guest pointers into the process's live mapped address space.
    unsafe { std::ptr::read_unaligned(address as *const u32) }
}

fn read_guest_u64(address: u64) -> u64 {
    // SAFETY: HLE calls use guest pointers into the process's live mapped address space.
    unsafe { std::ptr::read_unaligned(address as *const u64) }
}

fn write_guest_u32(address: u64, value: u32) {
    // SAFETY: HLE calls use guest pointers into the process's live mapped address space.
    unsafe {
        std::ptr::write_unaligned(address as *mut u32, value);
    }
}

fn write_guest_u16(address: u64, value: u16) {
    // SAFETY: HLE calls use guest pointers into the process's live mapped address space.
    unsafe {
        std::ptr::write_unaligned(address as *mut u16, value);
    }
}

fn write_output_u64(address: u64, value: u64) -> u64 {
    if address == 0 {
        return u64::MAX;
    }
    write_guest_u64(address, value);
    0
}

fn write_output_i32(address: u64, value: i32) -> u64 {
    if address == 0 {
        return u64::MAX;
    }
    // SAFETY: The HLE ABI identifies `address` as a writable 32-bit output field.
    unsafe {
        std::ptr::write_unaligned(address as *mut i32, value);
    }
    0
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

fn guest_memcpy(destination: u64, source: u64, length: u64) -> u64 {
    if length == 0 {
        return destination;
    }
    if destination == 0 || source == 0 {
        return 0;
    }
    let Ok(length) = usize::try_from(length) else {
        return 0;
    };
    // SAFETY: memcpy's guest ABI contract requires two valid non-overlapping ranges.
    unsafe {
        std::ptr::copy_nonoverlapping(source as *const u8, destination as *mut u8, length);
    }
    destination
}

fn guest_memmove(destination: u64, source: u64, length: u64) -> u64 {
    if length == 0 {
        return destination;
    }
    if destination == 0 || source == 0 {
        return 0;
    }
    let Ok(length) = usize::try_from(length) else {
        return 0;
    };
    // SAFETY: memmove's guest ABI contract supplies two valid ranges; `copy` permits overlap.
    unsafe {
        std::ptr::copy(source as *const u8, destination as *mut u8, length);
    }
    destination
}

fn guest_malloc(size: u64) -> u64 {
    let Ok(size) = usize::try_from(size) else {
        return 0;
    };
    // SAFETY: The returned allocation is process-local and uses the same x86-64 pointer ABI.
    unsafe { libc::malloc(size) as usize as u64 }
}

fn guest_free(address: u64) -> u64 {
    if address != 0 {
        // SAFETY: The guest passes an allocation previously returned by this libc HLE family.
        unsafe {
            libc::free(address as *mut std::ffi::c_void);
        }
    }
    0
}

fn guest_calloc(count: u64, size: u64) -> u64 {
    let (Ok(count), Ok(size)) = (usize::try_from(count), usize::try_from(size)) else {
        return 0;
    };
    // SAFETY: libc validates the allocation-size multiplication and returns process-local memory.
    unsafe { libc::calloc(count, size) as usize as u64 }
}

fn guest_realloc(address: u64, size: u64) -> u64 {
    let Ok(size) = usize::try_from(size) else {
        return 0;
    };
    // SAFETY: `address` is null or belongs to the same libc allocation family.
    unsafe { libc::realloc(address as *mut std::ffi::c_void, size) as usize as u64 }
}

fn guest_aligned_alloc(alignment: u64, size: u64) -> u64 {
    let (Ok(mut alignment), Ok(size)) = (usize::try_from(alignment), usize::try_from(size)) else {
        return 0;
    };
    if alignment == 0 || !alignment.is_power_of_two() {
        return 0;
    }
    // Guest memalign accepts sub-pointer alignments; the host allocator requires at least
    // sizeof(void*), which still satisfies the guest's weaker request.
    alignment = alignment.max(std::mem::size_of::<usize>());
    let mut output = std::ptr::null_mut();
    // SAFETY: `output` is writable and alignment/size are represented in the host ABI.
    let result = unsafe { libc::posix_memalign(&raw mut output, alignment, size) };
    if result == 0 {
        output as usize as u64
    } else {
        0
    }
}

fn guest_posix_memalign(output: u64, alignment: u64, size: u64) -> u64 {
    if output == 0 {
        return libc::EINVAL as u64;
    }
    let (Ok(alignment), Ok(size)) = (usize::try_from(alignment), usize::try_from(size)) else {
        return libc::ENOMEM as u64;
    };
    let mut allocation = std::ptr::null_mut();
    // SAFETY: `allocation` is writable and libc validates the requested alignment.
    let result = unsafe { libc::posix_memalign(&raw mut allocation, alignment, size) };
    if result == 0 {
        write_guest_u64(output, allocation as usize as u64);
    }
    u64::try_from(result).unwrap_or(u64::MAX)
}

fn guest_strlen(address: u64) -> u64 {
    if address == 0 {
        return 0;
    }
    // SAFETY: The guest libc call contract supplies a valid NUL-terminated string.
    unsafe { libc::strlen(address as *const libc::c_char) as u64 }
}

fn guest_strcmp(left: u64, right: u64) -> u64 {
    if left == 0 || right == 0 {
        return u64::MAX;
    }
    // SAFETY: strcmp's guest ABI contract supplies two NUL-terminated strings.
    let result = unsafe { libc::strcmp(left as *const libc::c_char, right as *const libc::c_char) };
    signed_i32_return(result)
}

fn guest_strncmp(left: u64, right: u64, length: u64) -> u64 {
    if left == 0 || right == 0 {
        return u64::MAX;
    }
    let Ok(length) = usize::try_from(length) else {
        return u64::MAX;
    };
    // SAFETY: strncmp's guest ABI contract supplies readable ranges through `length`.
    let result = unsafe {
        libc::strncmp(
            left as *const libc::c_char,
            right as *const libc::c_char,
            length,
        )
    };
    signed_i32_return(result)
}

fn guest_strcasecmp(left: u64, right: u64) -> u64 {
    if left == 0 || right == 0 {
        return u64::from(left != right);
    }
    // SAFETY: strcasecmp's guest ABI contract supplies two NUL-terminated strings.
    let result =
        unsafe { libc::strcasecmp(left as *const libc::c_char, right as *const libc::c_char) };
    signed_i32_return(result)
}

fn guest_fopen(path_address: u64, mode_address: u64, state: &ProbeDispatchState) -> u64 {
    let (Some(guest_path), Some(mode)) = (
        read_guest_cstring(path_address),
        read_guest_cstring(mode_address),
    ) else {
        return 0;
    };
    let Some(host_path) = resolve_guest_path(&state.game_root, &guest_path) else {
        eprintln!("[REMu][VFS] rejected path {guest_path:?}");
        return 0;
    };
    // The bring-up VFS is intentionally read-only; saves receive a separate writable mount later.
    if !mode.starts_with('r') {
        eprintln!("[REMu][VFS] rejected write mode {mode:?} for {guest_path:?}");
        return 0;
    }
    let (Ok(host_path), Ok(mode)) = (
        CString::new(host_path.to_string_lossy().as_bytes()),
        CString::new(mode),
    ) else {
        return 0;
    };
    // SAFETY: Both strings are NUL-terminated and host libc owns the returned FILE object.
    let file = unsafe { libc::fopen(host_path.as_ptr(), mode.as_ptr()) };
    eprintln!(
        "[REMu][VFS] fopen guest={guest_path:?} host={:?} result={file:p}",
        host_path.to_string_lossy()
    );
    file as usize as u64
}

fn resolve_guest_path(game_root: &Path, guest_path: &str) -> Option<PathBuf> {
    let normalized = guest_path.replace('\\', "/");
    let relative = if let Some(path) = normalized.strip_prefix("/app0/") {
        path
    } else if normalized == "/app0" {
        ""
    } else if let Some((_, path)) = normalized.split_once("/app0/") {
        path
    } else if normalized.starts_with('/') {
        return None;
    } else {
        normalized.as_str()
    };
    let mut host = game_root.to_owned();
    // Console paths routinely contain duplicate separators (for example
    // `/app0//shaders/presenter.vert`). Treat them like POSIX does while keeping parent traversal
    // outside the read-only game root impossible.
    for component in relative.split('/') {
        match component {
            "" | "." => {}
            ".." => return None,
            component => host.push(component),
        }
    }
    Some(host)
}

fn read_guest_cstring(address: u64) -> Option<String> {
    if address == 0 {
        return None;
    }
    // SAFETY: Guest libc string arguments are NUL-terminated within live guest memory.
    let string = unsafe { std::ffi::CStr::from_ptr(address as *const libc::c_char) };
    string.to_str().ok().map(ToOwned::to_owned)
}

fn guest_fseek(handle: u64, offset: u64, whence: u64) -> u64 {
    if handle == 0 {
        return u64::MAX;
    }
    // SAFETY: fopen returned the handle, and libc validates offset/whence.
    signed_i32_return(unsafe {
        libc::fseek(
            handle as *mut libc::FILE,
            i64::from_ne_bytes(offset.to_ne_bytes()),
            low_i32(whence),
        )
    })
}

fn guest_ftell(handle: u64) -> u64 {
    if handle == 0 {
        return u64::MAX;
    }
    // SAFETY: fopen returned the live FILE handle.
    unsafe { libc::ftell(handle as *mut libc::FILE).cast_unsigned() }
}

fn guest_fclose(handle: u64) -> u64 {
    if handle == 0 {
        return u64::MAX;
    }
    // SAFETY: fopen returned the live FILE handle and ownership ends here.
    signed_i32_return(unsafe { libc::fclose(handle as *mut libc::FILE) })
}

fn guest_fread(destination: u64, size: u64, count: u64, handle: u64) -> u64 {
    if destination == 0 || handle == 0 || size == 0 || count == 0 {
        return 0;
    }
    let (Ok(size), Ok(count)) = (usize::try_from(size), usize::try_from(count)) else {
        return 0;
    };
    // SAFETY: The guest supplies a destination sized for size*count and fopen returned the handle.
    unsafe {
        libc::fread(
            destination as *mut libc::c_void,
            size,
            count,
            handle as *mut libc::FILE,
        ) as u64
    }
}

fn guest_strcpy(destination: u64, source: u64) -> u64 {
    if destination == 0 || source == 0 {
        return 0;
    }
    // SAFETY: strcpy's guest ABI contract supplies a large-enough destination and C string source.
    unsafe {
        libc::strcpy(
            destination as *mut libc::c_char,
            source as *const libc::c_char,
        );
    }
    destination
}

fn guest_strncpy(destination: u64, source: u64, length: u64) -> u64 {
    if destination == 0 || source == 0 {
        return 0;
    }
    let Ok(length) = usize::try_from(length) else {
        return 0;
    };
    // SAFETY: strncpy's guest ABI contract supplies readable/writable ranges through `length`.
    unsafe {
        libc::strncpy(
            destination as *mut libc::c_char,
            source as *const libc::c_char,
            length,
        );
    }
    destination
}

fn guest_strcat(destination: u64, source: u64) -> u64 {
    if destination == 0 || source == 0 {
        return 0;
    }
    // SAFETY: strcat's guest ABI contract supplies a large-enough destination and C string source.
    unsafe {
        libc::strcat(
            destination as *mut libc::c_char,
            source as *const libc::c_char,
        );
    }
    destination
}

fn signed_i32_return(value: i32) -> u64 {
    u64::from_ne_bytes(i64::from(value).to_ne_bytes())
}

fn guest_usleep(microseconds: u64) -> u64 {
    let bounded = microseconds.min(u64::from(u32::MAX));
    let Ok(bounded) = u32::try_from(bounded) else {
        return u64::MAX;
    };
    // SAFETY: usleep has no pointer arguments and the duration is bounded to its host ABI width.
    signed_i32_return(unsafe { libc::usleep(bounded) })
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
    game_root: &Path,
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
    configure_probe_dispatch(image, game_root, maximum_import_hits);
    set_current_guest_thread_pointer(image.tls.thread_pointer.get());
    // SAFETY: Alarm only bounds this diagnostic process; the default SIGALRM action terminates it.
    unsafe {
        libc::alarm(10);
    }
    // SAFETY: The linker validated and mapped the ELF entry in executable guest memory. The process
    // entry ABI is the same System V x86-64 convention used by this Rosetta process.
    let entry: GuestEntry = unsafe { std::mem::transmute(image.memory.entry_address().get()) };
    // SAFETY: `parameters` and its image-name pointer stay live for the duration of the call.
    let return_value = unsafe { entry(&raw const parameters, guest_exit) };
    set_current_guest_thread_pointer(0);
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
    _game_root: &Path,
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

#[cfg(test)]
mod platform_independent_tests {
    use std::path::Path;

    use super::resolve_guest_path;

    #[test]
    fn resolves_duplicate_console_path_separators() {
        assert_eq!(
            resolve_guest_path(Path::new("/games/title"), "/app0//shaders/presenter.vert"),
            Some(Path::new("/games/title/shaders/presenter.vert").to_owned())
        );
    }

    #[test]
    fn rejects_game_root_parent_traversal() {
        assert_eq!(
            resolve_guest_path(Path::new("/games/title"), "/app0/../outside.bin"),
            None
        );
    }
}

#[cfg(all(test, target_os = "macos", target_arch = "x86_64"))]
mod tests {
    use std::io::Cursor;

    use remu_core::GuestAddress;
    use remu_loader::load_reader;

    use super::{LinkedImage, set_current_guest_thread_pointer};

    type TlsHelper = unsafe extern "C" fn() -> u64;

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
        let displacement = i32::from_le_bytes(
            patched_tls_load[1..5]
                .try_into()
                .expect("fixed-width call displacement"),
        );
        let helper_address = (base + 0x4145).wrapping_add_signed(i64::from(displacement));
        set_current_guest_thread_pointer(0x1234_5678_9abc_def0);
        // SAFETY: The linker emitted and finalized the tested helper as executable x86-64 code.
        let helper: TlsHelper = unsafe { std::mem::transmute(helper_address) };
        // SAFETY: The helper takes no arguments and returns the configured pointer in RAX.
        assert_eq!(unsafe { helper() }, 0x1234_5678_9abc_def0);
        set_current_guest_thread_pointer(0);
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
