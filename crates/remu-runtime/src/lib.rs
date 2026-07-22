// Copyright (C) 2026 REmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

mod agc_command_abi;
mod agc_register_defaults;
mod audio;
#[cfg(target_os = "macos")]
mod kernel_memory;
mod np;

use std::cell::Cell;
use std::collections::{BTreeMap, BTreeSet, HashMap, VecDeque};
use std::ffi::CString;
use std::io::{Read, Seek, Write};
#[cfg(unix)]
use std::os::unix::fs::{MetadataExt, OpenOptionsExt};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicUsize, Ordering};
use std::sync::{Arc, Condvar, Mutex, OnceLock, RwLock};
use std::thread::ThreadId;
use std::time::{Duration, Instant};

use remu_core::GuestAddress;
use remu_loader::{
    DynamicInfo, Elf64Rela, Elf64Symbol, ImportedSymbol, InspectError, LoadedExecutable,
    ModuleDependencyGraphError, UnresolvedModuleDependency, resolve_module_dependency_graph,
};
#[cfg(target_os = "macos")]
use remu_memory::{AddressSpaceError, GUEST_PAGE_SIZE, GuestMapPlacement};
use remu_memory::{GUEST_EXECUTE, GUEST_READ, GUEST_WRITE, GuestImage, MappingError};
use remu_shader::GuestMemoryReader;
use thiserror::Error;

#[cfg(target_os = "macos")]
#[link(name = "System")]
unsafe extern "C" {
    fn mach_task_self() -> u32;
    fn mach_vm_read_overwrite(
        task: u32,
        address: u64,
        size: u64,
        output_address: u64,
        output_size: *mut u64,
    ) -> i32;
}

const R_X86_64_64: u32 = 1;
const R_X86_64_GLOB_DAT: u32 = 6;
const R_X86_64_JUMP_SLOT: u32 = 7;
const R_X86_64_RELATIVE: u32 = 8;
const R_X86_64_DTPMOD64: u32 = 16;
const IMPORT_STUB_SIZE: u64 = 32;
const IMPORT_STUB_SIZE_USIZE: usize = 32;
const PT_TLS: u32 = 7;
const PT_SCE_PROC_PARAM: u32 = 0x6100_0001;
const PT_GNU_EH_FRAME: u32 = 0x6474_e550;
const DW_EH_PE_OMIT: u8 = 0xff;
const DW_EH_PE_INDIRECT: u8 = 0x80;
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
pub struct ImportResolutionStats {
    pub bundled_guest: usize,
    pub host_direct: usize,
    pub hle_dispatch: usize,
    pub unresolved_weak: usize,
}

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub enum ImportResolutionKind {
    BundledGuest,
    HostDirect,
    HleDispatch,
    UnresolvedWeak,
}

impl ImportResolutionKind {
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::BundledGuest => "bundled-guest",
            Self::HostDirect => "host-direct",
            Self::HleDispatch => "hle-dispatch",
            Self::UnresolvedWeak => "unresolved-weak",
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ImportResolutionRecord {
    pub symbol: String,
    pub nid: String,
    pub weak: bool,
    pub kind: ImportResolutionKind,
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

impl ImportResolutionStats {
    #[must_use]
    pub const fn total(self) -> usize {
        self.bundled_guest + self.host_direct + self.hle_dispatch + self.unresolved_weak
    }
}

#[derive(Debug)]
pub struct LinkedImage {
    pub memory: GuestImage,
    pub import_stubs: Vec<ImportStub>,
    pub relocation_stats: RelocationStats,
    pub tls: GuestTls,
    pub process_param: Option<GuestAddress>,
    import_resolution_stats: ImportResolutionStats,
    import_resolutions: Vec<ImportResolutionRecord>,
    unwind_fdes: Vec<GuestAddress>,
    #[cfg_attr(not(target_arch = "x86_64"), allow(dead_code))]
    module_initializers: Vec<GuestAddress>,
}

/// Filesystem and dependency information retained from linking a complete guest process image.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ModuleLinkReport {
    pub app_root: PathBuf,
    pub entry_path: PathBuf,
    /// Bundled PRX/SPRX objects in the dependency-first order used for mapping and initialization.
    pub loaded_modules: Vec<PathBuf>,
    /// Dependencies that must be supplied by the emulator's system-module implementations.
    pub unresolved_dependencies: Vec<UnresolvedModuleDependency>,
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
    #[error(transparent)]
    Inspect(#[from] InspectError),
    #[error(transparent)]
    ModuleGraph(#[from] ModuleDependencyGraphError),
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
    #[error("dynamic symbol {symbol:?} refers to an unknown Sony library or module namespace")]
    InvalidSymbolNamespace { symbol: String },
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
    #[error("unsupported guest .eh_frame pointer encoding 0x{0:02x}")]
    UnsupportedEhFrameEncoding(u8),
    #[error("malformed guest .eh_frame data: {0}")]
    MalformedEhFrame(&'static str),
    #[error("guest {kind} byte size 0x{size:x} is not a multiple of eight")]
    InvalidInitializerArraySize { kind: &'static str, size: u64 },
    #[error("guest initializer address {address} is outside the process image")]
    InitializerOutsideImage { address: GuestAddress },
    #[error("native macOS C++ unwind runtime is unavailable")]
    NativeCxxRuntimeUnavailable,
    #[error("native macOS unwind symbol {0} is unavailable")]
    MissingNativeUnwindSymbol(&'static str),
}

impl LinkedImage {
    /// Discovers, loads, maps, and relocates the complete bundled dependency graph for `path`.
    ///
    /// Missing system PRXs remain explicit in the returned report because those dependencies are
    /// fulfilled by `REMu`'s HLE services rather than files from the game dump.
    pub fn link_path(path: impl AsRef<Path>) -> Result<(Self, ModuleLinkReport), LinkError> {
        let graph = resolve_module_dependency_graph(path)?;
        let executable = remu_loader::load_path(&graph.entry.path)?;
        let mut modules = Vec::with_capacity(graph.dependency_order.len());
        for node in &graph.dependency_order {
            let name = node
                .original_filename
                .clone()
                .or_else(|| {
                    node.path
                        .file_name()
                        .and_then(|name| name.to_str())
                        .map(str::to_owned)
                })
                .unwrap_or_else(|| node.path.display().to_string());
            modules.push((name, remu_loader::load_path(&node.path)?));
        }
        let module_refs = modules
            .iter()
            .map(|(name, executable)| (name.as_str(), executable))
            .collect::<Vec<_>>();
        let image = Self::link_main_with_modules(&executable, &module_refs)?;
        let report = ModuleLinkReport {
            app_root: graph.app_root,
            entry_path: graph.entry.path,
            loaded_modules: graph
                .dependency_order
                .into_iter()
                .map(|node| node.path)
                .collect(),
            unresolved_dependencies: graph.unresolved_dependencies,
        };
        Ok((image, report))
    }

    pub fn link_main(executable: &LoadedExecutable) -> Result<Self, LinkError> {
        Self::link_main_with_modules(executable, &[])
    }

    /// Links the main executable and dependency-ordered game-bundled PRX modules into one process
    /// address space and symbol scope.
    pub fn link_main_with_modules(
        executable: &LoadedExecutable,
        modules: &[(&str, &LoadedExecutable)],
    ) -> Result<Self, LinkError> {
        let elf = executable.info.elf.as_ref().ok_or(LinkError::MissingElf)?;
        let dynamic = elf
            .dynamic
            .as_ref()
            .ok_or(LinkError::MissingDynamicMetadata)?;
        let auxiliary_reserve = calculate_process_auxiliary_reserve(dynamic, modules)?;
        let mut memory =
            GuestImage::map_main_with_auxiliary_reserve(executable, auxiliary_reserve)?;
        let linked_modules = map_bundled_modules(&mut memory, modules)?;

        let main_export = main_base_export_input(executable, dynamic, memory.image_base());
        let process_exports = collect_process_exports(&memory, &main_export, &linked_modules)?;
        let (import_resolution_stats, import_resolutions) = collect_import_resolutions(
            std::iter::once(dynamic).chain(linked_modules.iter().map(|module| module.dynamic)),
            &process_exports,
        )?;
        let mut import_stubs = Vec::new();
        let mut relocation_stats = RelocationStats::default();
        let mut unwind_fdes = Vec::new();
        for module in &linked_modules {
            let (module_stubs, symbol_stubs) = create_import_stubs(
                &mut memory,
                module.dynamic,
                import_stubs.len(),
                &process_exports,
            )?;
            import_stubs.extend(module_stubs);
            add_relocation_stats(
                &mut relocation_stats,
                apply_relocations_at(
                    &mut memory,
                    module.base,
                    module.tls_module_id,
                    module.dynamic,
                    &symbol_stubs,
                    &process_exports,
                )?,
            );
            unwind_fdes.extend(collect_unwind_fdes_at(
                &memory,
                module.base,
                module.executable,
            )?);
        }

        let (main_stubs, symbol_stubs) =
            create_import_stubs(&mut memory, dynamic, import_stubs.len(), &process_exports)?;
        import_stubs.extend(main_stubs);
        let main_base = memory.image_base();
        let process_param = resolve_process_param(elf, main_base)?;
        add_relocation_stats(
            &mut relocation_stats,
            apply_relocations_at(
                &mut memory,
                main_base,
                1,
                dynamic,
                &symbol_stubs,
                &process_exports,
            )?,
        );
        unwind_fdes.extend(collect_unwind_fdes_at(&memory, main_base, executable)?);
        let mut module_initializers = collect_initializer_array(
            &memory,
            main_base,
            dynamic.preinit_array_address,
            dynamic.preinit_array_size,
            "DT_PREINIT_ARRAY",
        )?;
        for module in &linked_modules {
            append_object_initializers(
                &mut module_initializers,
                &memory,
                module.base,
                module.dynamic,
            )?;
        }
        append_object_initializers(&mut module_initializers, &memory, main_base, dynamic)?;
        let tls_inputs = std::iter::once(TlsImageInput {
            executable,
            base: main_base,
            module_id: 1,
        })
        .chain(linked_modules.iter().map(|module| TlsImageInput {
            executable: module.executable,
            base: module.base,
            module_id: module.tls_module_id,
        }))
        .collect::<Vec<_>>();
        let tls = install_tls_support(&mut memory, &tls_inputs)?;
        memory.finalize_permissions()?;
        Ok(Self {
            memory,
            import_stubs,
            relocation_stats,
            tls,
            process_param,
            import_resolution_stats,
            import_resolutions,
            unwind_fdes,
            module_initializers,
        })
    }

    pub fn probe_entry(
        &self,
        process_image_name: &str,
        game_root: &Path,
        maximum_import_hits: usize,
        timeout_seconds: u32,
    ) -> Result<ProbeOutcome, LinkError> {
        probe_entry(
            self,
            process_image_name,
            game_root,
            maximum_import_hits,
            timeout_seconds,
        )
    }

    #[must_use]
    pub fn unwind_record_count(&self) -> usize {
        self.unwind_fdes.len()
    }

    #[must_use]
    pub fn import_resolution_stats(&self) -> ImportResolutionStats {
        self.import_resolution_stats
    }

    #[must_use]
    pub fn import_resolutions(&self) -> &[ImportResolutionRecord] {
        &self.import_resolutions
    }
}

fn resolve_process_param(
    elf: &remu_loader::Elf64Info,
    image_base: GuestAddress,
) -> Result<Option<GuestAddress>, LinkError> {
    elf.program_headers
        .iter()
        .find(|header| header.program_type == PT_SCE_PROC_PARAM)
        .map(|header| {
            image_base
                .get()
                .checked_add(header.virtual_address.get())
                .map(GuestAddress::new)
                .ok_or(LinkError::AddressOverflow)
        })
        .transpose()
}

struct LinkedModuleInput<'a> {
    executable: &'a LoadedExecutable,
    dynamic: &'a DynamicInfo,
    base: GuestAddress,
    tls_module_id: u64,
}

fn map_bundled_modules<'a>(
    memory: &mut GuestImage,
    modules: &[(&'a str, &'a LoadedExecutable)],
) -> Result<Vec<LinkedModuleInput<'a>>, LinkError> {
    let mut linked_modules = Vec::with_capacity(modules.len());
    for (index, (_name, module)) in modules.iter().enumerate() {
        let dynamic = module
            .info
            .elf
            .as_ref()
            .ok_or(LinkError::MissingElf)?
            .dynamic
            .as_ref()
            .ok_or(LinkError::MissingDynamicMetadata)?;
        let base = memory.map_module(module, MACOS_GUEST_PAGE_ALIGNMENT)?;
        linked_modules.push(LinkedModuleInput {
            executable: module,
            dynamic,
            base,
            tls_module_id: u64::try_from(index + 2).map_err(|_| LinkError::AddressOverflow)?,
        });
    }
    Ok(linked_modules)
}

fn calculate_process_auxiliary_reserve(
    main_dynamic: &DynamicInfo,
    modules: &[(&str, &LoadedExecutable)],
) -> Result<u64, LinkError> {
    let mut cursor = 0_u64;
    let mut import_count = main_dynamic.imports.len();
    for (_, module) in modules {
        cursor =
            align_up_u64(cursor, MACOS_GUEST_PAGE_ALIGNMENT).ok_or(LinkError::AddressOverflow)?;
        let module_size = module
            .load_segments
            .iter()
            .try_fold(0_u64, |maximum, segment| {
                segment
                    .virtual_address
                    .get()
                    .checked_add(segment.memory_size)
                    .map(|end| maximum.max(end))
                    .ok_or(LinkError::AddressOverflow)
            })?;
        if module_size == 0 {
            return Err(LinkError::Mapping(MappingError::NoLoadSegments));
        }
        cursor = cursor
            .checked_add(module_size)
            .ok_or(LinkError::AddressOverflow)?;
        let dynamic = module
            .info
            .elf
            .as_ref()
            .ok_or(LinkError::MissingElf)?
            .dynamic
            .as_ref()
            .ok_or(LinkError::MissingDynamicMetadata)?;
        import_count = import_count
            .checked_add(dynamic.imports.len())
            .ok_or(LinkError::StubAllocationOverflow)?;
    }

    cursor = align_up_u64(cursor, IMPORT_STUB_SIZE).ok_or(LinkError::AddressOverflow)?;
    let maximum_stub_bytes = u64::try_from(import_count)
        .map_err(|_| LinkError::StubAllocationOverflow)?
        .checked_mul(IMPORT_STUB_SIZE)
        .ok_or(LinkError::StubAllocationOverflow)?;
    cursor = cursor
        .checked_add(maximum_stub_bytes)
        .ok_or(LinkError::AddressOverflow)?;
    cursor = align_up_u64(cursor, 16).ok_or(LinkError::AddressOverflow)?;
    cursor = cursor
        .checked_add(TLS_HELPER_REGION_SIZE)
        .ok_or(LinkError::AddressOverflow)?;
    cursor = align_up_u64(cursor, MACOS_GUEST_PAGE_ALIGNMENT).ok_or(LinkError::AddressOverflow)?;
    cursor = cursor
        .checked_add(GUEST_TLS_PREFIX_SIZE + GUEST_TLS_TCB_SIZE)
        .ok_or(LinkError::AddressOverflow)?;
    // Keep one guard-sized cushion for alignment changes without hiding unbounded allocations.
    cursor = cursor
        .checked_add(MACOS_GUEST_PAGE_ALIGNMENT)
        .ok_or(LinkError::AddressOverflow)?;
    align_up_u64(cursor, MACOS_GUEST_PAGE_ALIGNMENT).ok_or(LinkError::AddressOverflow)
}

#[derive(Clone, Copy)]
struct TlsImageInput<'a> {
    executable: &'a LoadedExecutable,
    base: GuestAddress,
    module_id: u64,
}

fn add_relocation_stats(total: &mut RelocationStats, addition: RelocationStats) {
    total.absolute += addition.absolute;
    total.global_data += addition.global_data;
    total.jump_slots += addition.jump_slots;
    total.relative += addition.relative;
}

fn append_object_initializers(
    output: &mut Vec<GuestAddress>,
    memory: &GuestImage,
    image_base: GuestAddress,
    dynamic: &DynamicInfo,
) -> Result<(), LinkError> {
    if dynamic.init_address != 0 {
        output.push(normalize_process_address(
            memory,
            image_base,
            dynamic.init_address,
        )?);
    }
    output.extend(collect_initializer_array(
        memory,
        image_base,
        dynamic.init_array_address,
        dynamic.init_array_size,
        "DT_INIT_ARRAY",
    )?);
    Ok(())
}

fn collect_initializer_array(
    memory: &GuestImage,
    image_base: GuestAddress,
    address: u64,
    size: u64,
    kind: &'static str,
) -> Result<Vec<GuestAddress>, LinkError> {
    if address == 0 || size == 0 {
        return Ok(Vec::new());
    }
    if !size.is_multiple_of(8) {
        return Err(LinkError::InvalidInitializerArraySize { kind, size });
    }
    let count = usize::try_from(size / 8).map_err(|_| LinkError::AddressOverflow)?;
    let byte_length = usize::try_from(size).map_err(|_| LinkError::AddressOverflow)?;
    let array = runtime_address_at(memory, image_base, GuestAddress::new(address), byte_length)?;
    let mut bytes = vec![0_u8; byte_length];
    memory.read(array, &mut bytes)?;
    let mut initializers = Vec::with_capacity(count);
    for entry in bytes.chunks_exact(8) {
        let value = u64::from_le_bytes(entry.try_into().expect("fixed initializer pointer"));
        if value != 0 && value != u64::MAX {
            initializers.push(normalize_process_address(memory, image_base, value)?);
        }
    }
    Ok(initializers)
}

fn normalize_process_address(
    memory: &GuestImage,
    image_base: GuestAddress,
    address: u64,
) -> Result<GuestAddress, LinkError> {
    let address = GuestAddress::new(address);
    if memory.contains(address, 1) {
        return Ok(address);
    }
    let rebased = GuestAddress::new(
        image_base
            .get()
            .checked_add(address.get())
            .ok_or(LinkError::AddressOverflow)?,
    );
    if memory.contains(rebased, 1) {
        Ok(rebased)
    } else {
        Err(LinkError::InitializerOutsideImage { address: rebased })
    }
}

#[derive(Clone, Debug)]
struct ProcessExport {
    address: u64,
    weak: bool,
}

#[derive(Clone, Debug, Eq, Ord, PartialEq, PartialOrd)]
enum ProcessSymbolKey {
    Scoped {
        nid: String,
        library: String,
        library_version: u16,
        module: String,
        module_version_major: u8,
        module_version_minor: u8,
        symbol_type: u8,
    },
    Unscoped {
        nid: String,
        symbol_type: u8,
    },
}

impl ProcessSymbolKey {
    fn nid(&self) -> &str {
        match self {
            Self::Scoped { nid, .. } | Self::Unscoped { nid, .. } => nid,
        }
    }
}

fn main_base_export_input<'a>(
    executable: &'a LoadedExecutable,
    dynamic: &'a DynamicInfo,
    base: GuestAddress,
) -> LinkedModuleInput<'a> {
    LinkedModuleInput {
        executable,
        dynamic,
        base,
        tls_module_id: 1,
    }
}

fn collect_process_exports(
    memory: &GuestImage,
    main: &LinkedModuleInput<'_>,
    modules: &[LinkedModuleInput<'_>],
) -> Result<BTreeMap<ProcessSymbolKey, u64>, LinkError> {
    let mut exports = BTreeMap::<ProcessSymbolKey, ProcessExport>::new();
    for module in std::iter::once(main).chain(modules) {
        for symbol in module.dynamic.symbols.iter().filter(|symbol| {
            !symbol.is_undefined() && !symbol.name.is_empty() && matches!(symbol.binding(), 1 | 2)
        }) {
            let key = process_symbol_key(module.dynamic, symbol)?;
            if key.nid().is_empty() {
                continue;
            }
            let address = defined_symbol_address_at(memory, module.base, symbol)?;
            let candidate = ProcessExport {
                address,
                weak: symbol.is_weak(),
            };
            match exports.get_mut(&key) {
                None => {
                    exports.insert(key, candidate);
                }
                Some(current) if current.weak && !candidate.weak => {
                    *current = candidate;
                }
                Some(_) => {}
            }
        }
    }
    Ok(exports
        .into_iter()
        .map(|(nid, export)| (nid, export.address))
        .collect())
}

fn process_symbol_key(
    dynamic: &DynamicInfo,
    symbol: &Elf64Symbol,
) -> Result<ProcessSymbolKey, LinkError> {
    let mut ids = symbol.name.split('#');
    let nid = ids.next().unwrap_or_default();
    let library_id = ids.next();
    let module_id = ids.next();
    if ids.next().is_none()
        && let (Some(library_id), Some(module_id)) = (library_id, module_id)
        && let Some(library) = dynamic
            .imported_libraries
            .iter()
            .chain(&dynamic.exported_libraries)
            .find(|library| encode_sce_id(library.id) == library_id)
        && let Some(module) = dynamic
            .required_modules
            .iter()
            .chain(&dynamic.exported_modules)
            .find(|module| encode_sce_id(module.id) == module_id)
    {
        return Ok(ProcessSymbolKey::Scoped {
            nid: nid.to_owned(),
            library: library.name.clone(),
            library_version: library.version,
            module: module.name.clone(),
            module_version_major: module.version_major,
            module_version_minor: module.version_minor,
            symbol_type: symbol.symbol_type(),
        });
    }
    if library_id.is_some()
        && module_id.is_some()
        && (!dynamic.imported_libraries.is_empty()
            || !dynamic.exported_libraries.is_empty()
            || !dynamic.required_modules.is_empty()
            || !dynamic.exported_modules.is_empty())
    {
        return Err(LinkError::InvalidSymbolNamespace {
            symbol: symbol.name.clone(),
        });
    }
    Ok(ProcessSymbolKey::Unscoped {
        nid: nid.to_owned(),
        symbol_type: symbol.symbol_type(),
    })
}

fn encode_sce_id(id: u16) -> String {
    const CODES: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+-";
    let mut encoded = String::with_capacity(3);
    if id < 0x40 {
        encoded.push(char::from(CODES[usize::from(id)]));
    } else if id < 0x1000 {
        encoded.push(char::from(CODES[usize::from((id >> 6) & 0x3f)]));
        encoded.push(char::from(CODES[usize::from(id & 0x3f)]));
    } else {
        encoded.push(char::from(CODES[usize::from((id >> 12) & 0x3f)]));
        encoded.push(char::from(CODES[usize::from((id >> 6) & 0x3f)]));
        encoded.push(char::from(CODES[usize::from(id & 0x3f)]));
    }
    encoded
}

fn imported_symbol<'a>(
    dynamic: &'a DynamicInfo,
    import: &ImportedSymbol,
) -> Result<&'a Elf64Symbol, LinkError> {
    let index = usize::try_from(import.symbol_index).map_err(|_| LinkError::MissingSymbol {
        index: import.symbol_index,
    })?;
    dynamic.symbols.get(index).ok_or(LinkError::MissingSymbol {
        index: import.symbol_index,
    })
}

fn classify_import(
    dynamic: &DynamicInfo,
    import: &ImportedSymbol,
    process_exports: &BTreeMap<ProcessSymbolKey, u64>,
) -> Result<ImportResolutionKind, LinkError> {
    let symbol = imported_symbol(dynamic, import)?;
    let key = process_symbol_key(dynamic, symbol)?;
    if process_exports.contains_key(&key) {
        return Ok(ImportResolutionKind::BundledGuest);
    }
    let nid = key.nid();
    if native_import_target(nid).is_some() || synthetic_data_import_target(nid).is_some() {
        return Ok(ImportResolutionKind::HostDirect);
    }
    if import.weak {
        Ok(ImportResolutionKind::UnresolvedWeak)
    } else {
        Ok(ImportResolutionKind::HleDispatch)
    }
}

fn collect_import_resolutions<'a>(
    dynamics: impl IntoIterator<Item = &'a DynamicInfo>,
    process_exports: &BTreeMap<ProcessSymbolKey, u64>,
) -> Result<(ImportResolutionStats, Vec<ImportResolutionRecord>), LinkError> {
    let mut stats = ImportResolutionStats::default();
    let mut records = Vec::new();
    for dynamic in dynamics {
        for import in &dynamic.imports {
            let kind = classify_import(dynamic, import, process_exports)?;
            match kind {
                ImportResolutionKind::BundledGuest => stats.bundled_guest += 1,
                ImportResolutionKind::HostDirect => stats.host_direct += 1,
                ImportResolutionKind::HleDispatch => stats.hle_dispatch += 1,
                ImportResolutionKind::UnresolvedWeak => stats.unresolved_weak += 1,
            }
            let symbol = imported_symbol(dynamic, import)?;
            let key = process_symbol_key(dynamic, symbol)?;
            records.push(ImportResolutionRecord {
                symbol: symbol.name.clone(),
                nid: key.nid().to_owned(),
                weak: import.weak,
                kind,
            });
        }
    }
    records.sort_by(|left, right| {
        (&left.symbol, left.kind, left.weak).cmp(&(&right.symbol, right.kind, right.weak))
    });
    Ok((stats, records))
}

#[cfg(test)]
mod linker_metadata_tests {
    use remu_core::GuestAddress;
    use remu_loader::{DynamicInfo, Elf64Symbol, SceLibraryReference, SceModuleReference};

    use super::{ProcessSymbolKey, encode_sce_id, process_symbol_key};

    #[test]
    fn encodes_sony_namespace_ids_in_the_dynamic_symbol_alphabet() {
        assert_eq!(encode_sce_id(0), "A");
        assert_eq!(encode_sce_id(63), "-");
        assert_eq!(encode_sce_id(64), "BA");
        assert_eq!(encode_sce_id(0x0fff), "--");
        assert_eq!(encode_sce_id(0x1000), "BAA");
        assert_eq!(encode_sce_id(u16::MAX), "P--");
    }

    #[test]
    fn resolves_symbols_by_namespace_versions_and_type_not_local_encoded_ids() {
        let imported = dynamic_namespace(false, 4, 3, 7, 2, 1);
        let exported = dynamic_namespace(true, 11, 10, 7, 2, 1);
        let import_symbol = symbol("someNid#E#D", 0x12, 0);
        let export_symbol = symbol("someNid#L#K", 0x12, 1);

        let import_key = process_symbol_key(&imported, &import_symbol).unwrap();
        let export_key = process_symbol_key(&exported, &export_symbol).unwrap();
        assert_eq!(import_key, export_key);
        assert!(matches!(import_key, ProcessSymbolKey::Scoped { .. }));

        let different_version = dynamic_namespace(true, 11, 10, 8, 2, 1);
        assert_ne!(
            process_symbol_key(&different_version, &export_symbol).unwrap(),
            export_key
        );
        let object_export = symbol("someNid#L#K", 0x11, 1);
        assert_ne!(
            process_symbol_key(&exported, &object_export).unwrap(),
            export_key
        );
        assert!(process_symbol_key(&exported, &symbol("someNid#Z#Z", 0x12, 1)).is_err());
    }

    fn symbol(name: &str, info: u8, section_index: u16) -> Elf64Symbol {
        Elf64Symbol {
            name_offset: 0,
            info,
            other: 0,
            section_index,
            value: GuestAddress::new(0x100),
            size: 8,
            name: name.to_owned(),
        }
    }

    fn dynamic_namespace(
        exports: bool,
        library_id: u16,
        module_id: u16,
        library_version: u16,
        module_version_major: u8,
        module_version_minor: u8,
    ) -> DynamicInfo {
        let library = SceLibraryReference {
            name: "GuestLibrary".to_owned(),
            id: library_id,
            version: library_version,
        };
        let module = SceModuleReference {
            name: "GuestModule".to_owned(),
            id: module_id,
            version_major: module_version_major,
            version_minor: module_version_minor,
        };
        let (required_modules, exported_modules, imported_libraries, exported_libraries) =
            if exports {
                (Vec::new(), vec![module], Vec::new(), vec![library])
            } else {
                (vec![module], Vec::new(), vec![library], Vec::new())
            };
        DynamicInfo {
            entries: Vec::new(),
            needed_libraries: Vec::new(),
            original_filename: None,
            required_modules,
            exported_modules,
            imported_libraries,
            exported_libraries,
            string_table_address: 0,
            string_table_size: 0,
            symbol_table_address: 0,
            symbol_table_size: 0,
            rela_address: 0,
            rela_size: 0,
            jump_relocation_address: 0,
            jump_relocation_size: 0,
            plt_got_address: 0,
            init_address: 0,
            init_array_address: 0,
            init_array_size: 0,
            preinit_array_address: 0,
            preinit_array_size: 0,
            symbols: Vec::new(),
            relocations: Vec::new(),
            jump_relocations: Vec::new(),
            imports: Vec::new(),
        }
    }
}

/// Returns the number of guest pthread workers that are still executing.
///
/// A PS5 process can return from its entry routine after handing long-lived work to pthreads. The
/// graphical launcher uses this count to keep the mapped image and GPU event sink alive until those
/// workers finish or the launch timeout expires.
#[must_use]
pub fn active_guest_thread_count() -> usize {
    PROBE_DISPATCH.get().map_or(0, |state| {
        state
            .read()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .thread_handles
            .values()
            .filter(|thread| !thread.is_finished())
            .count()
    })
}

fn install_tls_support(
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
    // DTV generation, maximum module ID, and one two-word entry per TLS-bearing module.
    write_u64(memory, dtv, 1)?;
    let maximum_module_id = module_blocks
        .iter()
        .map(|(module_id, _)| *module_id)
        .max()
        .unwrap_or(0);
    write_u64(memory, add_address(dtv, 8)?, maximum_module_id)?;
    for (module_id, block) in module_blocks {
        let entry_offset = module_id
            .checked_mul(16)
            .ok_or(LinkError::AddressOverflow)?;
        write_u64(memory, add_address(dtv, entry_offset)?, block.get())?;
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

#[cfg(all(target_os = "macos", target_arch = "x86_64"))]
#[derive(Clone, Copy, Debug)]
struct NativeCxxRuntime {
    cxxabi_handle: usize,
    unwind_handle: usize,
}

#[cfg(all(target_os = "macos", target_arch = "x86_64"))]
fn native_cxx_runtime() -> Option<NativeCxxRuntime> {
    static RUNTIME: OnceLock<Option<NativeCxxRuntime>> = OnceLock::new();
    *RUNTIME.get_or_init(|| {
        // SAFETY: Both paths are NUL-terminated constants naming macOS shared-cache libraries.
        let cxxabi_handle = unsafe {
            libc::dlopen(
                c"/usr/lib/libc++abi.dylib".as_ptr(),
                libc::RTLD_NOW | libc::RTLD_LOCAL,
            )
        };
        // SAFETY: Same as above; the handle is intentionally retained for the process lifetime.
        let unwind_handle = unsafe {
            libc::dlopen(
                c"/usr/lib/system/libunwind.dylib".as_ptr(),
                libc::RTLD_NOW | libc::RTLD_LOCAL,
            )
        };
        if cxxabi_handle.is_null() || unwind_handle.is_null() {
            None
        } else {
            Some(NativeCxxRuntime {
                cxxabi_handle: cxxabi_handle as usize,
                unwind_handle: unwind_handle as usize,
            })
        }
    })
}

#[cfg(all(target_os = "macos", target_arch = "x86_64"))]
impl NativeCxxRuntime {
    fn cxxabi_symbol(self, name: &std::ffi::CStr) -> Option<u64> {
        // SAFETY: `cxxabi_handle` is a retained live dlopen handle and `name` is NUL-terminated.
        let address =
            unsafe { libc::dlsym(self.cxxabi_handle as *mut libc::c_void, name.as_ptr()) };
        (!address.is_null()).then_some(address as usize as u64)
    }

    fn unwind_symbol(self, name: &std::ffi::CStr) -> Option<u64> {
        // SAFETY: `unwind_handle` is a retained live dlopen handle and `name` is NUL-terminated.
        let address =
            unsafe { libc::dlsym(self.unwind_handle as *mut libc::c_void, name.as_ptr()) };
        (!address.is_null()).then_some(address as usize as u64)
    }
}

#[cfg(all(target_os = "macos", target_arch = "x86_64"))]
fn native_cxx_symbol_name(nid: &str) -> Option<&'static std::ffi::CStr> {
    Some(match nid {
        "XwLA5cTHjt4" => c"__gxx_personality_v0",
        "IJKVjsmxxWI" => c"__cxa_allocate_dependent_exception",
        "cfAXurvfl5o" => c"__cxa_allocate_exception",
        "pBxafllkvt0" => c"__cxa_bad_cast",
        "xcc6DTcL8QA" => c"__cxa_bad_typeid",
        "3cUUypQzMiI" => c"__cxa_begin_catch",
        "BxPeH9TTcs4" => c"__cxa_current_exception_type",
        "lX+4FNUklF0" => c"__cxa_end_catch",
        "kBxt5LwtLA4" => c"__cxa_free_dependent_exception",
        "nOIEswYD4Ig" => c"__cxa_free_exception",
        "Y6Sl4Xw7gfA" => c"__cxa_get_exception_ptr",
        "3GPpjQdAMTw" => c"__cxa_guard_acquire",
        "9rAeANT2tyE" => c"__cxa_guard_release",
        "2emaaluWzUw" => c"__cxa_guard_abort",
        "zr094EQ39Ww" => c"__cxa_pure_virtual",
        "ZL9FV4mJXxo" => c"__cxa_rethrow",
        "vkuuLfhnSZI" => c"__cxa_throw",
        "XbAd9Ach9mo" => c"__cxa_throw_bad_array_new_length",
        "cmBRQsLPqwA" => c"__cxa_uncaught_exception",
        "qYhnoevd9bI" => c"_ZSt9terminatev",
        "ozMAr28BwSY" => c"_ZSt14_Xout_of_rangePKc",
        "eT2UsmTewbU" => c"_ZSt11_Xbad_allocv",
        "pZ9WXcClPO8" => c"_ZTVN10__cxxabiv120__si_class_type_infoE",
        "byV+FWlAnB4" => c"_ZTVN10__cxxabiv117__class_type_infoE",
        "tQIo+GIPklo" => c"_ZSt14_Xlength_errorPKc",
        "9ByRMdo7ywg" => c"_ZTVN10__cxxabiv121__vmi_class_type_infoE",
        "V23qt24VPVs" => c"_ZSt17iostream_categoryv",
        "tyHd3P7oDrU" => c"_ZNKSt9exception8_DoraiseEv",
        "Q1BL70XVV0o" => c"_ZSt18uncaught_exceptionv",
        "MELi-cKqWq0" => c"_ZSt19_Xbad_function_callv",
        "NU-T4QowTNA" => c"_ZSt18_Xinvalid_argumentPKc",
        "hMAe+TWS9mQ" => c"__dynamic_cast",
        "apPZ6HKZWaQ" => c"_ZNKSt9exception4whatEv",
        "7P1Wm-5KgAY" => c"_ZNSt12domain_errorD1Ev",
        "MOBxtefPZUg" => c"_ZNSt9exceptionD2Ev",
        "5BIbzIuDxTQ" => c"_ZTISt12domain_error",
        "oAidKrxuUv0" => c"_ZTVSt12domain_error",
        "udTM6Nxx-Ng" => c"_ZTVSt11logic_error",
        "N2f485TmJms" => c"_ZNSt8ios_base7failureD1Ev",
        "oe9tS0VztYk" => c"_ZNSt13runtime_errorD2Ev",
        "47RvLSo2HN8" => c"_ZNSt8bad_castD1Ev",
        "oQDS9nX05Qg" => c"_ZNSt16invalid_argumentD1Ev",
        "Kcb+MNSzZcc" => c"_ZNSt12out_of_rangeD1Ev",
        "BJCgW9-OxLA" => c"_ZTISt8ios_base",
        "Bq8m04PN1zw" => c"_ZTVSt12system_error",
        "-L+-8F0+gBc" => c"_ZTVSt13runtime_error",
        "bLPn1gfqSW8" => c"_ZTISt13runtime_error",
        "qOD-ksTkE08" => c"_ZTISt8bad_cast",
        "tVHE+C8vGXk" => c"_ZTVSt8bad_cast",
        "XZzWt0ygWdw" => c"_ZTISt16invalid_argument",
        "keXoyW-rV-0" => c"_ZTVSt16invalid_argument",
        "dKjhNUf9FBc" => c"_ZTISt12out_of_range",
        "n+aUKkC-3sI" => c"_ZTVSt12out_of_range",
        "n2kx+OmFUis" => c"_ZTISt9exception",
        "BgZcGDh7o9g" => c"_ZNSt9exceptionD1Ev",
        "dCzeFfg9WWI" => c"_ZTVSt9exception",
        _ => return None,
    })
}

#[cfg(all(target_os = "macos", target_arch = "x86_64"))]
fn native_unwind_symbol_name(nid: &str) -> Option<&'static std::ffi::CStr> {
    Some(match nid {
        "s62MgBhosjU" => c"_Unwind_Backtrace",
        "GslDM6l8E7U" => c"_Unwind_DeleteException",
        "go1LaO7f15k" => c"_Unwind_FindEnclosingFunction",
        "Jhba7poZWZ0" => c"_Unwind_Find_FDE",
        "zs817D8+hPA" => c"_Unwind_ForcedUnwind",
        "uKwYpqeUkzo" => c"_Unwind_GetBSP",
        "MmTshn7gIgU" => c"_Unwind_GetCFA",
        "nqD9bBu-W6U" => c"_Unwind_GetDataRelBase",
        "oQpl5qCT014" => c"_Unwind_GetGR",
        "sETNbyWsEHs" => c"_Unwind_GetIP",
        "5-3wdp3FVtc" => c"_Unwind_GetIPInfo",
        "gpc4YXZffkQ" => c"_Unwind_GetLanguageSpecificData",
        "p3BVovMPEE4" => c"_Unwind_GetRegionStart",
        "HPlmfRhMWwo" => c"_Unwind_GetTextRelBase",
        "wpNJwmDDtxw" => c"_Unwind_RaiseException",
        "f1zwJ3jAI2k" => c"_Unwind_Resume",
        "xUsJSLsdv9I" => c"_Unwind_Resume_or_Rethrow",
        "Xr4pio56GGQ" => c"_Unwind_SetGR",
        "BWAtsvaXToA" => c"_Unwind_SetIP",
        _ => return None,
    })
}

#[cfg(all(target_os = "macos", target_arch = "x86_64"))]
fn native_libc_symbol_name(nid: &str) -> Option<&'static std::ffi::CStr> {
    Some(match nid {
        "8zTFvBIAIN8" => c"memset",
        "Q3VBxCXhUHs" => c"memcpy",
        "+P6FRGH4LfA" => c"memmove",
        "DfivPArhucg" => c"memcmp",
        "8u8lPzUEq+U" => c"memchr",
        "j4ViWNHEgww" => c"strlen",
        "Ovb2dSJOAuE" => c"strcmp",
        "aesyjrHVWy4" => c"strncmp",
        "kiZSXIWd9vg" => c"strcpy",
        "6sJWiWSRuqk" => c"strncpy",
        "Ls4tzzhimqQ" => c"strcat",
        "AV6ipCNa4Rw" => c"strcasecmp",
        "pXvbDfchu6k" => c"strncasecmp",
        "viiwFMaNamA" => c"strstr",
        "g7zzzLDYGw0" => c"strdup",
        "SfQIZcqvvms" => c"strlcpy",
        "5OqszGpy7Mg" => c"strtoull",
        "VOBg+iNwB-4" => c"strtoll",
        "2vDqwBlpF-o" => c"strtod",
        "dhK16CKwhQg" => c"__isfinited",
        "GfxAp9Xyiqs" => c"__isnand",
        "n88vx3C5nW8" => c"gettimeofday",
        "eLdDw6l0-bU" => c"snprintf",
        "Q2V+iqvjgC0" => c"vsnprintf",
        "tcVi5SivF7Q" => c"sprintf",
        "1Pk0qZQGeWo" => c"sscanf",
        "hcuQgD53UxM" => c"printf",
        "YQ0navp+YIc" => c"puts",
        "HUbZmOnT-Dg" => c"atan2",
        "5ZkEP3Rq7As" => c"cbrt",
        "JBcgYuW8lPU" => c"acos",
        "7Ly52zaL44Q" => c"asin",
        "OXmauLdQ8kY" => c"atan",
        "T7uyNqP7vQA" => c"tan",
        "kn0yiYeExgA" => c"ldexpf",
        "weDug8QD-lE" => c"atanf",
        "wuAQt-j+p4o" => c"exp2f",
        "hsi9drzHR2k" => c"log2f",
        "1D0H2KNjshE" => c"powf",
        "nlaojL9hDtA" => c"round",
        "DDHG1a6+3q0" => c"roundf",
        "9LCjpWyQ5Zc" => c"pow",
        "jMB7EFyu30Y" => c"sincos",
        "pKwslsMUmSk" => c"fmod",
        "RQXLbdT2lc4" => c"logf",
        "pztV4AF18iI" => c"sincosf",
        "H8ya2H00jbI" => c"sin",
        "Q4rRL34CEeE" => c"sinf",
        "rtV7-jWC6Yg" => c"log",
        "WuMbPBKN1TU" => c"log10",
        "2WE3BTYVwKM" => c"cos",
        "Y5DhuDKGlnQ" => c"log2",
        "NVadfnzQhHQ" => c"exp",
        "8zsu04XNsZ4" => c"expf",
        "-P6FNMzk2Kc" => c"cosf",
        _ => return None,
    })
}

#[cfg(all(target_os = "macos", target_arch = "x86_64"))]
fn native_import_target(nid: &str) -> Option<u64> {
    let runtime = native_cxx_runtime()?;
    if let Some(name) = native_cxx_symbol_name(nid) {
        return runtime.cxxabi_symbol(name);
    }
    if let Some(name) = native_unwind_symbol_name(nid) {
        return runtime.unwind_symbol(name);
    }
    runtime.cxxabi_symbol(native_libc_symbol_name(nid)?)
}

#[cfg(not(all(target_os = "macos", target_arch = "x86_64")))]
const fn native_import_target(_nid: &str) -> Option<u64> {
    None
}

extern "C" fn synthetic_data_noop(
    _argument0: u64,
    _argument1: u64,
    _argument2: u64,
    _argument3: u64,
    _argument4: u64,
    _argument5: u64,
) -> u64 {
    0
}

struct SyntheticDataImports {
    _classic_vtable: Box<[u64; 32]>,
    _classic_object: Box<[u64; 8]>,
    classic_symbol: Box<u64>,
    cells: BTreeMap<&'static str, Box<[u64; 32]>>,
}

fn synthetic_data_imports() -> &'static SyntheticDataImports {
    static IMPORTS: OnceLock<SyntheticDataImports> = OnceLock::new();
    IMPORTS.get_or_init(|| {
        let noop = synthetic_data_noop as *const () as usize as u64;
        let mut classic_vtable = Box::new([0_u64; 32]);
        classic_vtable[2..].fill(noop);
        let mut classic_object = Box::new([0_u64; 8]);
        classic_object[0] = classic_vtable.as_ptr() as usize as u64 + 16;
        let classic_symbol = Box::new(classic_object.as_ptr() as usize as u64);

        let mut cells = BTreeMap::new();
        for nid in [
            "H8AprKeZtNg",
            "sBCTjFk7Gi4",
            "yLE5H3058Ao",
            "Cv+zC4EjGMA",
            "H4fcpQOpc08",
            "FQ9NFbBHb5Y",
            "wiR+rIcbnlc",
            "E14mW8pVpoE",
            "1kZFcktOm+s",
            "VmqsS6auJzo",
            "irGo1yaJ-vM",
            "2sWzhYqFH4E",
            "f7uOxY9mM1U",
        ] {
            let mut cell = Box::new([0_u64; 32]);
            if nid == "FQ9NFbBHb5Y" {
                cell[0] = u64::MAX;
            } else if nid == "f7uOxY9mM1U" {
                cell[0] = 0x9e37_79b9_7f4a_7c15;
            } else if matches!(nid, "yLE5H3058Ao" | "1kZFcktOm+s") {
                cell[2..].fill(noop);
            }
            cells.insert(nid, cell);
        }
        SyntheticDataImports {
            _classic_vtable: classic_vtable,
            _classic_object: classic_object,
            classic_symbol,
            cells,
        }
    })
}

fn synthetic_data_import_target(nid: &str) -> Option<u64> {
    let imports = synthetic_data_imports();
    if nid == "Qoo175Ig+-k" {
        return Some((&raw const *imports.classic_symbol) as usize as u64);
    }
    imports
        .cells
        .get(nid)
        .map(|cell| cell.as_ptr() as usize as u64)
}

#[cfg(all(target_os = "macos", target_arch = "x86_64"))]
type RegisterFrame = unsafe extern "C" fn(*const libc::c_void);

#[cfg(all(target_os = "macos", target_arch = "x86_64"))]
fn register_guest_unwind_frames(image: &LinkedImage) -> Result<usize, LinkError> {
    if image.unwind_fdes.is_empty() {
        return Ok(0);
    }
    let runtime = native_cxx_runtime().ok_or(LinkError::NativeCxxRuntimeUnavailable)?;
    let register_address = runtime
        .unwind_symbol(c"__register_frame")
        .ok_or(LinkError::MissingNativeUnwindSymbol("__register_frame"))?;
    // SAFETY: dlsym returned the address of libunwind's documented single-pointer function.
    let register: RegisterFrame = unsafe {
        std::mem::transmute(
            usize::try_from(register_address).expect("x86-64 unwind address fits usize"),
        )
    };
    for fde in &image.unwind_fdes {
        // SAFETY: Every address was decoded and bounds-checked from the mapped guest .eh_frame.
        unsafe {
            register(
                usize::try_from(fde.get()).expect("x86-64 guest address fits usize")
                    as *const libc::c_void,
            );
        }
    }
    Ok(image.unwind_fdes.len())
}

fn create_import_stubs(
    memory: &mut GuestImage,
    dynamic: &DynamicInfo,
    first_dispatch_index: usize,
    process_exports: &BTreeMap<ProcessSymbolKey, u64>,
) -> Result<(Vec<ImportStub>, BTreeMap<u32, GuestAddress>), LinkError> {
    let mut dispatch_imports = BTreeMap::<ProcessSymbolKey, Vec<u32>>::new();
    for import in &dynamic.imports {
        if classify_import(dynamic, import, process_exports)? != ImportResolutionKind::HleDispatch {
            continue;
        }
        let symbol = imported_symbol(dynamic, import)?;
        dispatch_imports
            .entry(process_symbol_key(dynamic, symbol)?)
            .or_default()
            .push(import.symbol_index);
    }
    if dispatch_imports.is_empty() {
        return Ok((Vec::new(), BTreeMap::new()));
    }

    let stub_bytes = u64::try_from(dispatch_imports.len())
        .map_err(|_| LinkError::StubAllocationOverflow)?
        .checked_mul(IMPORT_STUB_SIZE)
        .ok_or(LinkError::StubAllocationOverflow)?;
    let stub_base =
        memory.allocate_staging(stub_bytes, IMPORT_STUB_SIZE, GUEST_READ | GUEST_EXECUTE)?;
    let bytes_len = usize::try_from(stub_bytes).map_err(|_| LinkError::StubAllocationOverflow)?;
    let mut bytes = vec![0_u8; bytes_len];
    let mut import_stubs = Vec::with_capacity(dispatch_imports.len());
    let mut symbol_stubs = BTreeMap::new();
    let dispatcher = import_dispatch_trampoline as *const () as usize as u64;
    for (index, (key, symbol_indices)) in dispatch_imports.into_iter().enumerate() {
        let byte_offset = index * IMPORT_STUB_SIZE_USIZE;
        let dispatch_index = first_dispatch_index
            .checked_add(index)
            .ok_or(LinkError::StubAllocationOverflow)?;
        write_import_stub(&mut bytes[byte_offset..], dispatch_index, dispatcher)?;
        let address = GuestAddress::new(
            stub_base
                .get()
                .checked_add(byte_offset as u64)
                .ok_or(LinkError::AddressOverflow)?,
        );
        for symbol_index in symbol_indices {
            symbol_stubs.insert(symbol_index, address);
        }
        import_stubs.push(ImportStub {
            nid: key.nid().to_owned(),
            address,
        });
    }
    memory.write(stub_base, &bytes)?;
    Ok((import_stubs, symbol_stubs))
}

fn apply_relocations_at(
    memory: &mut GuestImage,
    image_base: GuestAddress,
    tls_module_id: u64,
    dynamic: &DynamicInfo,
    symbol_stubs: &BTreeMap<u32, GuestAddress>,
    process_exports: &BTreeMap<ProcessSymbolKey, u64>,
) -> Result<RelocationStats, LinkError> {
    let mut stats = RelocationStats::default();
    for relocation in dynamic.relocations.iter().chain(&dynamic.jump_relocations) {
        let target = runtime_address_at(memory, image_base, relocation.offset, 8)?;
        let value = match relocation.relocation_type {
            R_X86_64_RELATIVE => {
                stats.relative += 1;
                image_base.get().wrapping_add_signed(relocation.addend)
            }
            R_X86_64_64 => {
                stats.absolute += 1;
                resolve_symbol_at(
                    memory,
                    image_base,
                    dynamic,
                    symbol_stubs,
                    process_exports,
                    relocation,
                )?
                .wrapping_add_signed(relocation.addend)
            }
            R_X86_64_GLOB_DAT => {
                stats.global_data += 1;
                resolve_symbol_at(
                    memory,
                    image_base,
                    dynamic,
                    symbol_stubs,
                    process_exports,
                    relocation,
                )?
            }
            R_X86_64_JUMP_SLOT => {
                stats.jump_slots += 1;
                resolve_symbol_at(
                    memory,
                    image_base,
                    dynamic,
                    symbol_stubs,
                    process_exports,
                    relocation,
                )?
            }
            R_X86_64_DTPMOD64 => tls_module_id,
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

fn resolve_symbol_at(
    memory: &GuestImage,
    image_base: GuestAddress,
    dynamic: &DynamicInfo,
    symbol_stubs: &BTreeMap<u32, GuestAddress>,
    process_exports: &BTreeMap<ProcessSymbolKey, u64>,
    relocation: &Elf64Rela,
) -> Result<u64, LinkError> {
    let index = usize::try_from(relocation.symbol_index).map_err(|_| LinkError::MissingSymbol {
        index: relocation.symbol_index,
    })?;
    let symbol = dynamic.symbols.get(index).ok_or(LinkError::MissingSymbol {
        index: relocation.symbol_index,
    })?;
    if symbol.is_undefined() {
        let key = process_symbol_key(dynamic, symbol)?;
        let nid = key.nid();
        if let Some(address) = process_exports.get(&key).copied() {
            return Ok(address);
        }
        if let Some(address) = native_import_target(nid) {
            return Ok(address);
        }
        if let Some(address) = synthetic_data_import_target(nid) {
            return Ok(address);
        }
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
    defined_symbol_address_at(memory, image_base, symbol)
}

fn defined_symbol_address_at(
    memory: &GuestImage,
    image_base: GuestAddress,
    symbol: &Elf64Symbol,
) -> Result<u64, LinkError> {
    if memory.contains(symbol.value, 1) {
        return Ok(symbol.value.get());
    }
    image_base
        .get()
        .checked_add(symbol.value.get())
        .ok_or(LinkError::AddressOverflow)
}

fn runtime_address_at(
    memory: &GuestImage,
    image_base: GuestAddress,
    value: GuestAddress,
    length: usize,
) -> Result<GuestAddress, LinkError> {
    if memory.contains(value, length) {
        return Ok(value);
    }
    Ok(GuestAddress::new(
        image_base
            .get()
            .checked_add(value.get())
            .ok_or(LinkError::AddressOverflow)?,
    ))
}

fn collect_unwind_fdes_at(
    memory: &GuestImage,
    image_base: GuestAddress,
    executable: &LoadedExecutable,
) -> Result<Vec<GuestAddress>, LinkError> {
    let Some(header) = executable
        .info
        .elf
        .as_ref()
        .and_then(|elf| {
            elf.program_headers
                .iter()
                .find(|header| header.program_type == PT_GNU_EH_FRAME)
        })
        .filter(|header| header.file_size >= 4)
    else {
        return Ok(Vec::new());
    };
    let header_size = usize::try_from(header.file_size)
        .map_err(|_| LinkError::MalformedEhFrame(".eh_frame_hdr is too large"))?;
    let header_address =
        runtime_address_at(memory, image_base, header.virtual_address, header_size)?;
    let mut header_bytes = vec![0_u8; header_size];
    memory.read(header_address, &mut header_bytes)?;
    if header_bytes[0] != 1 {
        return Err(LinkError::MalformedEhFrame(
            "unsupported .eh_frame_hdr version",
        ));
    }
    let mut cursor = 4;
    let frame_address = decode_eh_pointer(
        memory,
        &header_bytes,
        &mut cursor,
        header_bytes[1],
        header_address,
    )?;
    if !memory.contains(GuestAddress::new(frame_address), 4) {
        return Err(LinkError::MalformedEhFrame(
            ".eh_frame pointer is outside the process image",
        ));
    }
    let expected_fdes = if header_bytes[2] == DW_EH_PE_OMIT {
        0
    } else {
        usize::try_from(decode_eh_pointer(
            memory,
            &header_bytes,
            &mut cursor,
            header_bytes[2],
            header_address,
        )?)
        .map_err(|_| LinkError::MalformedEhFrame("FDE count is too large"))?
    };
    let table_encoding = header_bytes[3];
    if expected_fdes != 0 && table_encoding == DW_EH_PE_OMIT {
        return Err(LinkError::MalformedEhFrame(
            "nonempty .eh_frame_hdr omits its search table",
        ));
    }
    let mut fdes = Vec::with_capacity(expected_fdes);
    let mut unique_fdes = BTreeSet::new();
    for _ in 0..expected_fdes {
        let _initial_location = decode_eh_pointer(
            memory,
            &header_bytes,
            &mut cursor,
            table_encoding,
            header_address,
        )?;
        let fde = decode_eh_pointer(
            memory,
            &header_bytes,
            &mut cursor,
            table_encoding,
            header_address,
        )?;
        validate_fde(memory, fde)?;
        if unique_fdes.insert(fde) {
            fdes.push(GuestAddress::new(fde));
        }
    }
    if fdes.len() != expected_fdes {
        return Err(LinkError::MalformedEhFrame(
            ".eh_frame_hdr contains duplicate FDE pointers",
        ));
    }
    Ok(fdes)
}

fn decode_eh_pointer(
    memory: &GuestImage,
    bytes: &[u8],
    cursor: &mut usize,
    encoding: u8,
    data_relative_base: GuestAddress,
) -> Result<u64, LinkError> {
    if encoding == DW_EH_PE_OMIT {
        return Err(LinkError::UnsupportedEhFrameEncoding(encoding));
    }
    let field_address = data_relative_base
        .get()
        .checked_add(u64::try_from(*cursor).map_err(|_| LinkError::AddressOverflow)?)
        .ok_or(LinkError::AddressOverflow)?;
    let (raw, width, signed) = match encoding & 0x0f {
        0x00 | 0x04 => (read_eh_unsigned(bytes, *cursor, 8)?, 8, false),
        0x02 => (read_eh_unsigned(bytes, *cursor, 2)?, 2, false),
        0x03 => (read_eh_unsigned(bytes, *cursor, 4)?, 4, false),
        0x0a => (read_eh_unsigned(bytes, *cursor, 2)?, 2, true),
        0x0b => (read_eh_unsigned(bytes, *cursor, 4)?, 4, true),
        0x0c => (read_eh_unsigned(bytes, *cursor, 8)?, 8, true),
        _ => return Err(LinkError::UnsupportedEhFrameEncoding(encoding)),
    };
    *cursor = cursor
        .checked_add(width)
        .ok_or(LinkError::AddressOverflow)?;
    let value = if signed {
        sign_extend_eh_value(raw, width)
    } else {
        i128::from(raw)
    };
    let base = match encoding & 0x70 {
        0x00 => 0_i128,
        0x10 => i128::from(field_address),
        0x30 => i128::from(data_relative_base.get()),
        _ => return Err(LinkError::UnsupportedEhFrameEncoding(encoding)),
    };
    let resolved = base.checked_add(value).ok_or(LinkError::MalformedEhFrame(
        "encoded unwind pointer overflows the guest address space",
    ))?;
    let resolved = u64::try_from(resolved).map_err(|_| {
        LinkError::MalformedEhFrame("encoded unwind pointer overflows the guest address space")
    })?;
    if encoding & DW_EH_PE_INDIRECT == 0 {
        return Ok(resolved);
    }
    let mut indirect = [0_u8; 8];
    memory.read(GuestAddress::new(resolved), &mut indirect)?;
    Ok(u64::from_le_bytes(indirect))
}

fn read_eh_unsigned(bytes: &[u8], offset: usize, width: usize) -> Result<u64, LinkError> {
    let end = offset
        .checked_add(width)
        .ok_or(LinkError::AddressOverflow)?;
    let source = bytes.get(offset..end).ok_or(LinkError::MalformedEhFrame(
        "encoded unwind pointer exceeds .eh_frame_hdr",
    ))?;
    let mut value = [0_u8; 8];
    value[..width].copy_from_slice(source);
    Ok(u64::from_le_bytes(value))
}

fn sign_extend_eh_value(value: u64, width: usize) -> i128 {
    let shift = 64 - width * 8;
    i128::from((value << shift).cast_signed() >> shift)
}

fn validate_fde(memory: &GuestImage, address: u64) -> Result<(), LinkError> {
    let mut short_header = [0_u8; 12];
    memory.read(GuestAddress::new(address), &mut short_header)?;
    let short_length = u32::from_le_bytes(short_header[..4].try_into().expect("FDE length"));
    if short_length == 0 {
        return Err(LinkError::MalformedEhFrame("FDE has a zero length"));
    }
    let (length_field_size, entry_length, cie_pointer_size, cie_pointer) =
        if short_length == u32::MAX {
            (
                12_u64,
                u64::from_le_bytes(short_header[4..12].try_into().expect("extended FDE length")),
                8_usize,
                None,
            )
        } else {
            (
                4_u64,
                u64::from(short_length),
                4_usize,
                Some(u64::from(u32::from_le_bytes(
                    short_header[4..8].try_into().expect("FDE CIE pointer"),
                ))),
            )
        };
    if entry_length < cie_pointer_size as u64 {
        return Err(LinkError::MalformedEhFrame(
            "FDE is shorter than its CIE pointer",
        ));
    }
    let entry_size = length_field_size
        .checked_add(entry_length)
        .ok_or(LinkError::MalformedEhFrame("FDE size overflow"))?;
    let entry_size =
        usize::try_from(entry_size).map_err(|_| LinkError::MalformedEhFrame("FDE is too large"))?;
    if !memory.contains(GuestAddress::new(address), entry_size) {
        return Err(LinkError::MalformedEhFrame(
            "FDE extends outside the process image",
        ));
    }
    let cie_pointer = if let Some(pointer) = cie_pointer {
        pointer
    } else {
        let mut pointer = [0_u8; 8];
        let pointer_address = address
            .checked_add(length_field_size)
            .ok_or(LinkError::MalformedEhFrame("FDE address overflow"))?;
        memory.read(GuestAddress::new(pointer_address), &mut pointer)?;
        u64::from_le_bytes(pointer)
    };
    if cie_pointer == 0 {
        return Err(LinkError::MalformedEhFrame(
            ".eh_frame_hdr table points to a CIE instead of an FDE",
        ));
    }
    Ok(())
}

/// Returns the service name for NIDs whose ABI is currently implemented or under active bring-up.
#[must_use]
#[allow(clippy::too_many_lines)]
pub fn known_nid_name(nid: &str) -> Option<&'static str> {
    Some(match nid {
        "bzQExy189ZI" => "_init_env",
        "8G2LB+A3rzg" => "atexit",
        "tsvEmnenz48" => "__cxa_atexit",
        "H2e8t5ScQGc" => "__cxa_finalize",
        "rNhWz+lvOMU" => "_sceKernelSetThreadDtors",
        "pB-yGZ2nQ9o" => "_sceKernelSetThreadAtexitCount",
        "WhCc1w3EhSI" => "_sceKernelSetThreadAtexitReport",
        "959qrazPIrg" => "sceKernelGetProcParam",
        "p5EcQeEeJAE" => "_sceKernelRtldSetApplicationHeapAPI",
        "bnZxYgAFeA0" => "sceKernelGetSanitizerNewReplaceExternal",
        "NWtTN10cJzE" => "sceLibcHeapGetTraceInfo",
        "mkx2fVhNMsg" => "pthread_cond_broadcast",
        "QxqK-IdpumU" => "_Getpmbstate",
        "zS94yyJRSUs" => "_Getpwcstate",
        "-9SIhUr4Iuo" => "_Mbtowcx",
        "stv1S3BKfgw" => "_Wctombx",
        "ATpTPg64JhM" => "_Wctombx.initial",
        "FxVZqBAA7ks" => "_write",
        "4wSze92BhLI" => "sceKernelWrite",
        "aI+OeCz8xrQ" => "scePthreadSelf",
        "pO96TwzOm5E" => "sceKernelGetDirectMemorySize",
        "rTXw65xmLIA" => "sceKernelAllocateDirectMemory",
        "L-Q3LEjIbgA" => "sceKernelMapDirectMemory",
        "BQQniolj9tQ" => "sceKernelMapDirectMemory2",
        "hwVSPCmp5tM" => "sceKernelCheckedReleaseDirectMemory",
        "MBuItvba6z8" => "sceKernelReleaseDirectMemory",
        "rVjRvHJ0X6c" => "sceKernelVirtualQuery",
        "cQke9UuBQOk" => "sceKernelMunmap",
        "D0OdFMjp46I" => "sceKernelCreateEqueue",
        "fzyMKs9kim0" => "sceKernelWaitEqueue",
        "lLMT9vJAck0" => "clock_gettime",
        "n88vx3C5nW8" => "gettimeofday",
        "wLlFkwG9UcQ" => "time",
        "9BcDykPmo1I" => "__error",
        "5OqszGpy7Mg" => "strtoull",
        "zlfEH8FmyUA" => "_Stoul",
        "VOBg+iNwB-4" => "strtoll",
        "2vDqwBlpF-o" => "strtod",
        "8u8lPzUEq+U" => "memchr",
        "fL3O02ypZFE" => "wmemcpy",
        "Noj9PsJrsa8" => "wmemmove",
        "fnUEjBCNRVU" => "wmemchr",
        "QJ5xVfKkni0" => "wmemcmp",
        "9rMML086SEE" => "std::locale::_Init",
        "hEQ2Yi4PJXA" => "std::locale::_Getgloballocale",
        "dhK16CKwhQg" => "__isfinite",
        "cfAXurvfl5o" => "__cxa_allocate_exception",
        "nOIEswYD4Ig" => "__cxa_free_exception",
        "vkuuLfhnSZI" => "__cxa_throw",
        "F8bUHwAG284" => "scePthreadMutexattrInit",
        "smWEktiyyG0" => "scePthreadMutexattrDestroy",
        "iMp8QpE+XO4" => "scePthreadMutexattrSettype",
        "cmo1RIYva9o" => "scePthreadMutexInit",
        "9UK1vLZQft4" => "scePthreadMutexLock",
        "tn3VlD0hG60" => "scePthreadMutexUnlock",
        "2Of0f+3mhhE" => "scePthreadMutexDestroy",
        "7H0iTOciTLo" => "pthread_mutex_lock",
        "2Z+PpY6CaJg" => "pthread_mutex_unlock",
        "nsYoNRywwNg" => "scePthreadAttrInit",
        "62KCwEMmzcM" => "scePthreadAttrDestroy",
        "-Wreprtu0Qs" => "scePthreadAttrSetdetachstate",
        "UTXzJbWhhTE" => "scePthreadAttrSetstacksize",
        "DzES9hQF4f4" => "scePthreadAttrSetschedparam",
        "x1X76arYMxU" => "scePthreadAttrGet",
        "-quPa4SEJUw" => "scePthreadAttrGetstack",
        "8+s5BzZjxSg" => "scePthreadAttrGetaffinity",
        "m5-2bsNfv7s" => "scePthreadCondattrInit",
        "waPcxYiR3WA" => "scePthreadCondattrDestroy",
        "2Tb92quprl0" => "scePthreadCondInit",
        "BmMjYxmew1w" => "scePthreadCondTimedwait",
        "4qGrR6eoP9Y" => "scePthreadDetach",
        "T72hz6ffq08" => "scePthreadYield",
        "6UgtwV+0zb4" => "scePthreadCreate",
        "onNY9Byn-W8" => "scePthreadJoin",
        "188x57JYp0g" => "sceKernelCreateSema",
        "Zxa0VhQVTsk" => "sceKernelWaitSema",
        "4czppHBiriw" => "sceKernelSignalSema",
        "1jfXLRVzisc" => "sceKernelUsleep",
        "g8cM39EUZ6o" => "sceSysmoduleLoadModule",
        "hZIg1EWGsHM" => "sceUltInitialize",
        "CdWp0oHWGr0" => "sceUserServiceGetInitialUser",
        "fPhymKNvK-A" => "sceUserServiceGetLoginUserIdList",
        "j3YMu1MVNNo" => "sceUserServiceInitialize",
        "hv1luiJrqQM" => "scePadInit",
        "xk0AcarP3V4" => "scePadOpen",
        "clVvL4ZDntw" => "scePadSetMotionSensorState",
        "gjP9-KQzoUk" => "scePadGetControllerInformation",
        "YndgXqQVV7c" => "scePadReadState",
        "JfEPXVxhFqA" => "sceAudioOutInit",
        "ekNvsT22rsY" => "sceAudioOutOpen",
        "s1--uE9mBFw" => "sceAudioOutClose",
        "b+uAV89IlxE" => "sceAudioOutSetVolume",
        "QOQtbeDqsT4" => "sceAudioOutOutput",
        "GrQ9s4IrNaQ" => "sceAudioOutGetPortState",
        "rPo6tV8D9bM" => "sceSystemServiceGetStatus",
        "fZo48un7LK4" => "sceSystemServiceParamGetInt",
        "Vo5V8KAwCmk" => "sceSystemServiceHideSplashScreen",
        "uoUpLGNkygk" => "sceCommonDialogInitialize",
        "ERKzksauAJA" => "sceSaveDataDialogGetStatus",
        "sjaobBgqeB4" => "sceNpUniversalDataSystemInitialize",
        "47UAEuQl+iI" => "sceNpUniversalDataSystemTerminate",
        "5zBnau1uIEo" => "sceNpUniversalDataSystemCreateContext",
        "wB7IWzGp2v0" => "sceNpUniversalDataSystemDestroyContext",
        "hT0IAEvN+M0" => "sceNpUniversalDataSystemCreateHandle",
        "jZCqWFgMehE" => "sceNpUniversalDataSystemAbortHandle",
        "AUIHb7jUX3I" => "sceNpUniversalDataSystemDestroyHandle",
        "tpFJ8LIKvPw" => "sceNpUniversalDataSystemRegisterContext",
        "p+GcLqwpL9M" => "sceNpUniversalDataSystemCreateEvent",
        "wG+84pnNIuo" => "sceNpUniversalDataSystemDestroyEvent",
        "YE4dbtbz6OE" => "sceNpUniversalDataSystemEventPropertyObjectSetInt32",
        "MfDb+4Nln64" => "sceNpUniversalDataSystemEventPropertyObjectSetString",
        "Wxbg5x3pTXA" => "sceNpUniversalDataSystemEventPropertyObjectSetArray",
        "4llLk7YJRTE" => "sceNpUniversalDataSystemEventPropertyArraySetString",
        "CzkKf7ahIyU" => "sceNpUniversalDataSystemPostEvent",
        "m87BHxt-H60" => "sceNpGameIntentInitialize",
        "jEIXUAr9XE8" => "sceNpGameIntentReceiveIntent",
        "rPl0INNc-M8" => "sceNpGameIntentGetPropertyValueString",
        "Bagshr7OQ6Q" => "sceNpTrophy2CreateContext",
        "Gz1rmUZpROM" => "sceNpTrophy2CreateHandle",
        "sysY2FHYff4" => "sceNpTrophy2DestroyContext",
        "d8P11CI40KE" => "sceNpTrophy2DestroyHandle",
        "fYapWA9xVmA" => "sceNpTrophy2AbortHandle",
        "bIDov3wBu5Q" => "sceNpTrophy2RegisterContext",
        "sUXGfNMalIo" => "sceNpTrophy2RegisterUnlockCallback",
        "wVqxM58sIKs" => "sceNpTrophy2UnregisterUnlockCallback",
        "vSMAm3cxYTY" => "sceKernelMprotect",
        "23LRUSvYu1M" => "sceAgcInit",
        "f3dg2CSgRKY" => "sceAgcCreateShader",
        "TRO721eVt4g" => "sceAgcDcbResetQueue",
        "ZvwO9euwYzc" => "sceAgcDcbSetCxRegistersIndirect",
        "-HOOCn0JY48" => "sceAgcDcbSetShRegistersIndirect",
        "hvUfkUIQcOE" => "sceAgcDcbSetUcRegistersIndirect",
        "D9sr1xGUriE" => "sceAgcCreatePrimitiveState",
        "HV4j+E0MBHE" => "sceAgcCreateInterpolantMapping",
        "l4fM9K-Lyks" => "sceAgcDcbSetIndexBuffer",
        "GIIW2J37e70" => "sceAgcDcbSetIndexSize",
        "B+aG9DUnTKA" => "sceAgcDcbDrawIndexOffset",
        "tSBxhAPyytQ" => "sceAgcDcbSetNumInstances",
        "Yw0jKSqop+E" => "sceAgcDcbDrawIndexAuto",
        "n2fD4A+pb+g" => "sceAgcCbSetShRegisterRangeDirect",
        "V++UgBtQhn0" => "sceAgcGetDataPacketPayloadAddress",
        "qj7QZpgr9Uw" => "sceAgcUnknownQj7QZpgr9Uw",
        "vcmNN+AAXnY" => "sceAgcSetCxRegIndirectPatchSetAddress",
        "d-6uF9sZDIU" => "sceAgcSetCxRegIndirectPatchAddRegisters",
        "Qrj4c+61z4A" => "sceAgcSetShRegIndirectPatchSetAddress",
        "z2duB-hHQSM" => "sceAgcSetShRegIndirectPatchAddRegisters",
        "6lNcCp+fxi4" => "sceAgcSetUcRegIndirectPatchSetAddress",
        "vRoArM9zaIk" => "sceAgcSetUcRegIndirectPatchAddRegisters",
        "aJf+j5yntiU" => "sceAgcDcbEventWrite",
        "LtTouSCZjHM" => "sceAgcCbNop",
        "WmAc2MEj6Io" => "sceAgcDcbDmaData",
        "wr23dPKyWc0" => "sceAgcCbReleaseMem",
        "VmW0Tdpy420" => "sceAgcDcbWaitRegMem",
        "IxYiarKlXxM" => "sceAgcDmaDataPatchSetDstAddressOrOffset",
        "3KDcnM3lrcU" => "sceAgcWaitRegMemPatchAddress",
        "0fWWK5uG9rQ" => "sceAgcQueueEndOfPipeActionPatchAddress",
        "57labkp+rSQ" => "sceAgcDcbAcquireMem",
        "HXzjK9yI30k" => "sceVideoOutAddFlipEvent",
        "Up36PTk687E" => "sceVideoOutOpen",
        "PjS5uASwcV8" => "sceVideoOutSetBufferAttribute2",
        "rKBUtgRrtbk" => "sceVideoOutRegisterBuffers2",
        "UglJIZjGssM" => "sceAgcDriverSubmitDcb",
        "6UzEidRZwkg" => "sceAgcDriverSubmitMultiDcbs",
        "gSRnr79F8tQ" => "sceAgcDriverSubmitAcb",
        "HF3YllT3mXU" => "sceAgcDriverSubmitMultiAcbs",
        "YUeqkyT7mEQ" => "sceAgcDcbSetFlip",
        "h9z6+0hEydk" => "sceAgcSuspendPoint",
        "1G3lF1Gg1k8" => "sceKernelOpen",
        "eV9wAD2riIA" => "sceKernelStat",
        "Cg4srZ6TKbU" => "sceKernelRead",
        "UK2Tl2DWUns" => "sceKernelClose",
        "6c3rCVE-fTU" => "_open",
        "wuCroIGjt2g" => "open",
        "DRuBt2pvICk" => "read",
        "AqBioC2vF3I" => "posix_read",
        "NNtFaKJbPt0" => "close",
        "bY-PO6JhzhQ" => "posix_close",
        "FN4gaPmuFV8" => "posix_write",
        "Oy6IpwgtYOk" => "lseek",
        "oib76F-12fk" => "sceKernelLseek",
        "E6ao34wPw+U" => "stat",
        "mqQMh1zPPT8" => "fstat",
        "kBwCPsYX-m4" => "sceKernelFstat",
        "ih4CD9-gghM" => "ftruncate",
        "VW3TVZiM4-E" => "sceKernelFtruncate",
        "TywrFKCoLGY" => "sceSaveDataInitialize3",
        "dyIhnXq-0SM" => "sceSaveDataDirNameSearch",
        "ZP4e7rlzOUk" => "sceSaveDataMount3",
        "gjRZNnw0JPE" => "sceSaveDataCreateTransactionResource",
        "uW4vfTwMQVo" => "sceSaveDataUmount2",
        "sDCBrmc61XU" => "sceSaveDataPrepare",
        "85zul--eGXs" => "sceSaveDataSetParam",
        "c88Yy54Mx0w" => "sceSaveDataSaveIcon",
        "ie7qhZ4X0Cc" => "sceSaveDataCommit",
        "pztV4AF18iI" => "sincosf",
        "AEJdIVZTEmo" => "qsort",
        "sUP1hBaouOw" => "_Getpctype",
        "rcQCUr0EaRU" => "_Getptoupper",
        _ => return None,
    })
}

fn write_import_stub(slot: &mut [u8], index: usize, dispatcher: u64) -> Result<(), LinkError> {
    let index = i32::try_from(index).map_err(|_| LinkError::StubAllocationOverflow)?;
    // Push the dispatch index without altering the guest's argument registers, then tail-jump via
    // R11 into the shared frame-capture trampoline. The trampoline removes this word before
    // returning to the original guest call site.
    slot[0] = 0x68;
    slot[1..5].copy_from_slice(&index.to_le_bytes());
    slot[5] = 0x49;
    slot[6] = 0xbb;
    slot[7..15].copy_from_slice(&dispatcher.to_le_bytes());
    slot[15..18].copy_from_slice(&[0x41, 0xff, 0xe3]);
    Ok(())
}

#[derive(Debug, Default)]
struct ProbeDispatchState {
    unimplemented_nids: BTreeSet<String>,
    #[cfg(target_os = "macos")]
    kernel_memory: Option<kernel_memory::KernelMemory>,
    next_thread_unique_id: u64,
    thread_handles: BTreeMap<u64, std::thread::JoinHandle<u64>>,
    thread_attributes: BTreeMap<u64, PthreadAttrState>,
    detached_threads: BTreeSet<u64>,
    pthread_threads: BTreeMap<u64, PthreadThreadObject>,
    pthread_attributes: BTreeMap<u64, PthreadAttrObject>,
    pthread_mutex_attributes: BTreeMap<u64, PthreadMutexAttrObject>,
    pthread_condition_attributes: BTreeSet<u64>,
    exit_callbacks: Vec<GuestExitCallback>,
    process_param_address: u64,
    thread_atexit_count_callback: u64,
    thread_atexit_report_callback: u64,
    thread_dtors_callback: u64,
    application_heap_api_address: u64,
    heap_trace_storage: Vec<u64>,
    services: RuntimeServiceState,
    audio_out: audio::AudioOutState,
    agc_primary_register_defaults: u64,
    agc_internal_register_defaults: u64,
    ctype_table: u64,
    tolower_table: u64,
    toupper_table: u64,
    locale_object: u64,
    locale_vtable: u64,
    tls_template: Vec<u8>,
    video_out: VideoOutState,
    game_root: PathBuf,
    np_universal_data: np::UniversalDataState,
    np_game_intent: np::GameIntentState,
    np_trophy: np::TrophyState,
    agc_submission_count: u64,
    agc_graphics: AgcGraphicsState,
    agc_compute_queues: BTreeMap<u32, AgcGraphicsState>,
    agc_shader_registry: AgcShaderRegistry,
    agc_index_resources: BTreeMap<u64, u64>,
    pending_events: VecDeque<KernelQueuedEvent>,
    next_kernel_fd: i32,
    kernel_files: BTreeMap<i32, std::fs::File>,
    save_root: PathBuf,
    save_data_initialized: bool,
    save_mounts: BTreeMap<String, SaveDataMount>,
    next_save_transaction_resource: u32,
    save_transaction_resources: BTreeSet<u32>,
    prepared_save_resources: BTreeSet<u32>,
    save_transactions: BTreeMap<u32, SaveDataTransaction>,
}

const DEFAULT_THREAD_AFFINITY_MASK: u64 = 0x7f;
const DEFAULT_THREAD_STACK_SIZE: u64 = 0x10_0000;
const DEFAULT_THREAD_GUARD_SIZE: u64 = 0x1000;
const DEFAULT_THREAD_PRIORITY: i32 = 700;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct PthreadAttrState {
    affinity_mask: u64,
    detach_state: i32,
    stack_address: u64,
    stack_size: u64,
    guard_size: u64,
    inherit_schedule: i32,
    schedule_policy: i32,
    schedule_priority: i32,
}

impl Default for PthreadAttrState {
    fn default() -> Self {
        Self {
            affinity_mask: DEFAULT_THREAD_AFFINITY_MASK,
            detach_state: 0,
            stack_address: 0,
            stack_size: DEFAULT_THREAD_STACK_SIZE,
            guard_size: DEFAULT_THREAD_GUARD_SIZE,
            inherit_schedule: 4,
            schedule_policy: 1,
            schedule_priority: DEFAULT_THREAD_PRIORITY,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct PthreadAttrObject {
    handle: u64,
    attributes: PthreadAttrState,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct PthreadMutexAttrState {
    mutex_type: i32,
    protocol: i32,
}

impl Default for PthreadMutexAttrState {
    fn default() -> Self {
        // Gen5's default mutex attribute is error-checking rather than host-pthread's normal type.
        Self {
            mutex_type: 1,
            protocol: 0,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct PthreadMutexAttrObject {
    handle: u64,
    attributes: PthreadMutexAttrState,
}

const PTHREAD_THREAD_OBJECT_SIZE: u64 = 0x1000;

#[derive(Clone, Debug, Eq, PartialEq)]
struct PthreadThreadObject {
    handle: u64,
    unique_id: u64,
    name: String,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct GuestExitCallback {
    function: u64,
    argument: Option<u64>,
    module: u64,
}

#[derive(Debug, Default)]
struct RuntimeServiceState {
    libc_environment: Option<u64>,
    loaded_sysmodules: BTreeSet<i32>,
    ult_runtime: Option<u64>,
    splash: SplashState,
    user_service: InitializationState,
    pad: InitializationState,
    common_dialog: InitializationState,
    save_data_dialog_status: i32,
    agc_states: BTreeMap<u64, u32>,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
enum InitializationState {
    #[default]
    Uninitialized,
    Initialized,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
enum SplashState {
    #[default]
    Visible,
    Hidden,
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct SaveDataMount {
    path: PathBuf,
    committed_path: PathBuf,
    dir_name: String,
    user_id: i32,
    resource: u32,
    read_only: bool,
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct SaveDataTransaction {
    mount_point: String,
    committed_path: PathBuf,
    staging_path: PathBuf,
}

#[derive(Debug, Default)]
struct VideoOutState {
    is_open: bool,
    user_id: i32,
    bus_type: i32,
    index: i32,
    pixel_format: u64,
    tiling_mode: u32,
    width: u32,
    height: u32,
    display_buffers: Vec<u64>,
    display_buffer_attributes: Vec<Option<VideoOutBufferAttributes>>,
    flip_events: BTreeMap<u64, u64>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct VideoOutBufferAttributes {
    width: u32,
    height: u32,
    pitch: u32,
    pixel_format: u64,
    tiling_mode: u32,
}

#[derive(Debug, Default)]
struct AgcGraphicsState {
    context_registers: BTreeMap<u32, u32>,
    shader_registers: BTreeMap<u32, u32>,
    user_config_registers: BTreeMap<u32, u32>,
    index_buffer_address: u64,
    index_resource_address: u64,
    index_buffer_count: u32,
    index_size: u32,
    instance_count: u32,
    draw_index_offset: u32,
    compute_indirect_args_address: u64,
    emitted_draws: u64,
    emitted_compute_dispatches: u64,
    traced_shader_pairs: BTreeSet<(u64, u64)>,
    known_color_targets: BTreeMap<u64, remu_gpu::GpuColorTarget>,
    pending_targetless_draw: Option<Box<remu_gpu::IndexedDraw>>,
}

#[derive(Debug, Default)]
struct AgcShaderRegistry {
    headers: BTreeMap<u64, u64>,
    snapshots: BTreeMap<u64, remu_gpu::ShaderSnapshot>,
    decode_failures: BTreeSet<u64>,
}

impl AgcGraphicsState {
    fn reset_pipeline(&mut self) {
        self.context_registers.clear();
        self.shader_registers.clear();
        self.user_config_registers.clear();
        self.index_buffer_address = 0;
        self.index_resource_address = 0;
        self.index_buffer_count = 0;
        self.index_size = 0;
        self.instance_count = 1;
        self.draw_index_offset = 0;
        self.compute_indirect_args_address = 0;
        self.pending_targetless_draw = None;
    }
}

#[derive(Clone, Copy, Debug)]
struct KernelQueuedEvent {
    equeue: u64,
    ident: u64,
    filter: i16,
    flags: u16,
    fflags: u32,
    data: u64,
    user_data: u64,
}

static PROBE_DISPATCH: OnceLock<RwLock<ProbeDispatchState>> = OnceLock::new();

#[derive(Debug)]
struct ImportTelemetry {
    nids: Box<[String]>,
    seen: Box<[AtomicBool]>,
    counts: Box<[AtomicUsize]>,
    hits: AtomicUsize,
    maximum_hits: usize,
}

impl ImportTelemetry {
    fn new(nids: Vec<String>, maximum_hits: usize) -> Self {
        let count = nids.len();
        Self {
            nids: nids.into_boxed_slice(),
            seen: (0..count)
                .map(|_| AtomicBool::new(false))
                .collect::<Vec<_>>()
                .into_boxed_slice(),
            counts: (0..count)
                .map(|_| AtomicUsize::new(0))
                .collect::<Vec<_>>()
                .into_boxed_slice(),
            hits: AtomicUsize::new(0),
            maximum_hits,
        }
    }

    fn hot_imports(&self, limit: usize) -> Vec<(&str, usize)> {
        let mut counts = self
            .nids
            .iter()
            .zip(&self.counts)
            .map(|(nid, count)| (nid.as_str(), count.load(Ordering::Relaxed)))
            .collect::<Vec<_>>();
        counts.sort_unstable_by_key(|(_, count)| std::cmp::Reverse(*count));
        counts.truncate(limit);
        counts
    }
}

static IMPORT_TELEMETRY: OnceLock<RwLock<Arc<ImportTelemetry>>> = OnceLock::new();

#[derive(Debug, Default)]
struct GuestSyncState {
    mutexes: HashMap<u64, GuestMutexState>,
    conditions: HashMap<u64, u64>,
    semaphores: HashMap<u32, GuestSemaphoreState>,
    next_semaphore_handle: u32,
}

#[derive(Debug)]
struct GuestMutexState {
    owner: Option<ThreadId>,
    recursion: u32,
    mutex_type: i32,
}

impl Default for GuestMutexState {
    fn default() -> Self {
        let attributes = PthreadMutexAttrState::default();
        Self {
            owner: None,
            recursion: 0,
            mutex_type: attributes.mutex_type,
        }
    }
}

#[derive(Debug)]
struct GuestSemaphoreState {
    count: i32,
    maximum: i32,
}

static GUEST_SYNC: OnceLock<(Mutex<GuestSyncState>, Condvar)> = OnceLock::new();

thread_local! {
    static CURRENT_GUEST_THREAD_POINTER: Cell<u64> = const { Cell::new(0) };
    static CURRENT_GUEST_THREAD_HANDLE: Cell<u64> = const { Cell::new(0) };
}

extern "C" fn current_guest_thread_pointer() -> u64 {
    CURRENT_GUEST_THREAD_POINTER.get()
}

fn set_current_guest_thread_pointer(thread_pointer: u64) {
    CURRENT_GUEST_THREAD_POINTER.set(thread_pointer);
}

fn current_guest_thread_handle() -> u64 {
    CURRENT_GUEST_THREAD_HANDLE.get()
}

fn set_current_guest_thread_handle(thread_handle: u64) {
    CURRENT_GUEST_THREAD_HANDLE.set(thread_handle);
}

fn allocate_pthread_thread_object(state: &mut ProbeDispatchState, name: &str) -> Option<u64> {
    let unique_id = state.next_thread_unique_id.max(1);
    state.next_thread_unique_id = unique_id.checked_add(1)?;
    let handle = guest_calloc(1, PTHREAD_THREAD_OBJECT_SIZE);
    if handle == 0 {
        return None;
    }
    state.pthread_threads.insert(
        handle,
        PthreadThreadObject {
            handle,
            unique_id,
            name: if name.is_empty() {
                format!("Thread-{unique_id:X}")
            } else {
                name.to_owned()
            },
        },
    );
    Some(handle)
}

fn retire_detached_guest_thread(handle: u64) {
    let Some(dispatch) = PROBE_DISPATCH.get() else {
        return;
    };
    let object = {
        let mut state = dispatch
            .write()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        if !state.detached_threads.remove(&handle) {
            return;
        }
        state.thread_handles.remove(&handle);
        state.thread_attributes.remove(&handle);
        state.pthread_threads.remove(&handle)
    };
    if let Some(object) = object {
        guest_free(object.handle);
    }
}

#[cfg(target_arch = "x86_64")]
fn configure_probe_dispatch(image: &LinkedImage, game_root: &Path, maximum_hits: usize) {
    let state = PROBE_DISPATCH.get_or_init(|| RwLock::new(ProbeDispatchState::default()));
    let mut state = state
        .write()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let nids = image
        .import_stubs
        .iter()
        .map(|stub| stub.nid.clone())
        .collect();
    let telemetry =
        IMPORT_TELEMETRY.get_or_init(|| RwLock::new(Arc::new(ImportTelemetry::new(Vec::new(), 0))));
    *telemetry
        .write()
        .unwrap_or_else(std::sync::PoisonError::into_inner) =
        Arc::new(ImportTelemetry::new(nids, maximum_hits));
    state.unimplemented_nids.clear();
    #[cfg(target_os = "macos")]
    {
        state.kernel_memory = None;
    }
    state.thread_handles.clear();
    state.thread_attributes.clear();
    state.detached_threads.clear();
    for object in std::mem::take(&mut state.pthread_threads).into_values() {
        guest_free(object.handle);
    }
    state.next_thread_unique_id = 1;
    let main_thread = allocate_pthread_thread_object(&mut state, "main")
        .expect("the process main-thread object allocation must succeed");
    state
        .thread_attributes
        .insert(main_thread, PthreadAttrState::default());
    for object in std::mem::take(&mut state.pthread_attributes).into_values() {
        guest_free(object.handle);
    }
    for object in std::mem::take(&mut state.pthread_mutex_attributes).into_values() {
        guest_free(object.handle);
    }
    state.pthread_condition_attributes.clear();
    state.exit_callbacks.clear();
    state.process_param_address = image.process_param.map_or(0, GuestAddress::get);
    state.thread_atexit_count_callback = 0;
    state.thread_atexit_report_callback = 0;
    state.thread_dtors_callback = 0;
    state.application_heap_api_address = 0;
    state.heap_trace_storage.clear();
    state.heap_trace_storage.resize(65, 0);
    state.services = RuntimeServiceState::default();
    state.audio_out = audio::AudioOutState::default();
    state.agc_primary_register_defaults = 0;
    state.agc_internal_register_defaults = 0;
    state.ctype_table = 0;
    state.tolower_table = 0;
    state.toupper_table = 0;
    state.locale_object = 0;
    state.locale_vtable = 0;
    let tls_template_address = image.tls.thread_pointer.get() - GUEST_TLS_PREFIX_SIZE;
    let tls_template_size = usize::try_from(GUEST_TLS_PREFIX_SIZE + GUEST_TLS_TCB_SIZE)
        .expect("guest TLS template size fits usize");
    state.tls_template.resize(tls_template_size, 0);
    // SAFETY: The linked image owns this live TLS reservation for the entire probe. Snapshot it
    // before guest entry so later workers never inherit the main thread's mutated allocator caches.
    unsafe {
        std::ptr::copy_nonoverlapping(
            tls_template_address as *const u8,
            state.tls_template.as_mut_ptr(),
            tls_template_size,
        );
    }
    state.video_out = VideoOutState::default();
    game_root.clone_into(&mut state.game_root);
    state.np_universal_data = np::UniversalDataState::default();
    state.np_game_intent = np::GameIntentState::default();
    state.np_trophy = np::TrophyState::default();
    state.agc_submission_count = 0;
    state.agc_graphics = AgcGraphicsState::default();
    state.agc_compute_queues.clear();
    state.agc_shader_registry = AgcShaderRegistry::default();
    state.agc_index_resources.clear();
    state.pending_events.clear();
    state.next_kernel_fd = 3;
    state.kernel_files.clear();
    state.save_root = save_data_root(game_root);
    state.save_data_initialized = false;
    state.save_mounts.clear();
    state.next_save_transaction_resource = 1;
    state.save_transaction_resources.clear();
    state.prepared_save_resources.clear();
    state.save_transactions.clear();
    let sync = GUEST_SYNC.get_or_init(|| (Mutex::new(GuestSyncState::default()), Condvar::new()));
    *sync
        .0
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner) = GuestSyncState::default();
    set_current_guest_thread_handle(main_thread);
}

#[repr(C, align(16))]
#[derive(Clone, Copy, Debug, Default)]
struct ImportCallFrame {
    integer_arguments: [u64; 6],
    rax: u64,
    guest_rsp: u64,
    guest_return_address: u64,
    dispatch_index: u64,
    xmm_arguments: [u128; 8],
}

const IMPORT_CALL_FRAME_SIZE: usize = 0xd0;
const _: () = {
    assert!(std::mem::size_of::<ImportCallFrame>() == IMPORT_CALL_FRAME_SIZE);
    assert!(std::mem::align_of::<ImportCallFrame>() == 16);
    assert!(std::mem::offset_of!(ImportCallFrame, rax) == 0x30);
    assert!(std::mem::offset_of!(ImportCallFrame, guest_rsp) == 0x38);
    assert!(std::mem::offset_of!(ImportCallFrame, guest_return_address) == 0x40);
    assert!(std::mem::offset_of!(ImportCallFrame, dispatch_index) == 0x48);
    assert!(std::mem::offset_of!(ImportCallFrame, xmm_arguments) == 0x50);
};

#[cfg(target_arch = "x86_64")]
#[unsafe(naked)]
unsafe extern "C" fn import_dispatch_trampoline() {
    core::arch::naked_asm!(
        "sub rsp, 0xd0",
        "mov [rsp + 0x00], rdi",
        "mov [rsp + 0x08], rsi",
        "mov [rsp + 0x10], rdx",
        "mov [rsp + 0x18], rcx",
        "mov [rsp + 0x20], r8",
        "mov [rsp + 0x28], r9",
        "mov [rsp + 0x30], rax",
        "lea r10, [rsp + 0xd8]",
        "mov [rsp + 0x38], r10",
        "mov r10, [rsp + 0xd8]",
        "mov [rsp + 0x40], r10",
        "mov r10, [rsp + 0xd0]",
        "mov [rsp + 0x48], r10",
        "movdqu [rsp + 0x50], xmm0",
        "movdqu [rsp + 0x60], xmm1",
        "movdqu [rsp + 0x70], xmm2",
        "movdqu [rsp + 0x80], xmm3",
        "movdqu [rsp + 0x90], xmm4",
        "movdqu [rsp + 0xa0], xmm5",
        "movdqu [rsp + 0xb0], xmm6",
        "movdqu [rsp + 0xc0], xmm7",
        "mov rdi, rsp",
        "call {dispatch}",
        "add rsp, 0xd0",
        "add rsp, 8",
        "ret",
        dispatch = sym import_dispatch,
    );
}

#[cfg(all(test, target_arch = "x86_64"))]
const TEST_CAPTURE_DISPATCH_INDEX: u64 = i32::MAX as u64;

#[cfg(all(test, target_arch = "x86_64"))]
static TEST_CAPTURED_IMPORT_FRAME: Mutex<Option<ImportCallFrame>> = Mutex::new(None);

#[cfg(all(test, target_arch = "x86_64"))]
#[unsafe(naked)]
unsafe extern "C" fn invoke_test_import_stub(_stub: u64) -> ProbeImportReturn {
    core::arch::naked_asm!(
        "mov r11, rdi",
        "mov rdi, 0x11",
        "mov rsi, 0x22",
        "mov rdx, 0x33",
        "mov rcx, 0x44",
        "mov r8, 0x55",
        "mov r9, 0x66",
        "mov rax, 0x0102030405060708",
        "mov r10, 0x101",
        "movq xmm0, r10",
        "mov r10, 0x202",
        "movq xmm1, r10",
        "mov r10, 0x303",
        "movq xmm2, r10",
        "mov r10, 0x404",
        "movq xmm3, r10",
        "mov r10, 0x505",
        "movq xmm4, r10",
        "mov r10, 0x606",
        "movq xmm5, r10",
        "mov r10, 0x707",
        "movq xmm6, r10",
        "mov r10, 0x808",
        "movq xmm7, r10",
        "sub rsp, 8",
        "call r11",
        "add rsp, 8",
        "ret",
    );
}

#[cfg(not(target_arch = "x86_64"))]
extern "C" fn import_dispatch_trampoline() {
    std::hint::black_box(
        import_dispatch as extern "C" fn(*const ImportCallFrame) -> ProbeImportReturn,
    );
    unreachable!("x86-64 guest import stubs cannot execute on this host architecture");
}

fn low_xmm_f64(value: u128) -> f64 {
    let bytes = value.to_le_bytes();
    f64::from_bits(u64::from_le_bytes(
        bytes[..8].try_into().expect("fixed XMM lane"),
    ))
}

#[allow(clippy::too_many_lines)]
extern "C" fn import_dispatch(frame: *const ImportCallFrame) -> ProbeImportReturn {
    if frame.is_null() {
        eprintln!("[REMu][IMPORT] null captured register frame");
        terminate_probe(87);
    }
    // SAFETY: import_dispatch_trampoline supplies one aligned, fully initialized frame that stays
    // live until this call returns.
    let frame = unsafe { &*frame };
    #[cfg(all(test, target_arch = "x86_64"))]
    if frame.dispatch_index == TEST_CAPTURE_DISPATCH_INDEX {
        *TEST_CAPTURED_IMPORT_FRAME
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner) = Some(*frame);
        return ProbeImportReturn {
            integer: frame.rax,
            float: low_xmm_f64(frame.xmm_arguments[7]),
        };
    }
    let call_args = frame.integer_arguments;
    let mut stack_args = [0_u64; 6];
    for (argument, output) in stack_args.iter_mut().enumerate() {
        let offset = u64::try_from((argument + 1) * std::mem::size_of::<u64>())
            .expect("fixed stack argument offset fits u64");
        *output = read_guest_u64(frame.guest_rsp.saturating_add(offset));
    }
    let float_argument0 = low_xmm_f64(frame.xmm_arguments[0]);
    let index = frame.dispatch_index;
    let guest_return_address = frame.guest_return_address;
    let telemetry = IMPORT_TELEMETRY
        .get_or_init(|| RwLock::new(Arc::new(ImportTelemetry::new(Vec::new(), 0))))
        .read()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .clone();
    let Some(index) = usize::try_from(index).ok() else {
        eprintln!("[REMu][IMPORT] invalid stub index {index}");
        terminate_probe(87);
    };
    let Some(nid) = telemetry.nids.get(index).map(String::as_str) else {
        eprintln!("[REMu][IMPORT] invalid stub index {index}");
        terminate_probe(87);
    };
    let hit = telemetry.hits.fetch_add(1, Ordering::Relaxed) + 1;
    telemetry.counts[index].fetch_add(1, Ordering::Relaxed);
    if !telemetry.seen[index].swap(true, Ordering::Relaxed) {
        let name = known_nid_name(nid).unwrap_or("unknown");
        eprintln!(
            "[REMu][IMPORT] first_hit={} nid={} name={} return={guest_return_address:#x} args={:#x},{:#x},{:#x},{:#x},{:#x},{:#x}",
            hit,
            nid,
            name,
            call_args[0],
            call_args[1],
            call_args[2],
            call_args[3],
            call_args[4],
            call_args[5],
        );
    }
    if telemetry.maximum_hits != 0 && hit >= telemetry.maximum_hits {
        eprintln!("[REMu][PROBE] import hit limit reached at nid={nid}");
        for (nid, count) in telemetry.hot_imports(8) {
            eprintln!("[REMu][PROBE] hot_import nid={nid} hits={count}");
        }
        terminate_probe(86);
    }
    if matches!(nid, "uMei1W9uyNo" | "6Z83sYWFlA8") {
        let status = low_i32(call_args[0]);
        let run_callbacks = nid == "uMei1W9uyNo";
        if run_callbacks {
            run_all_guest_exit_callbacks();
        }
        eprintln!(
            "[REMu][PROBE] guest requested {}({status})",
            if run_callbacks { "exit" } else { "_exit" }
        );
        terminate_probe(status);
    }
    if nid == "T72hz6ffq08" {
        std::thread::yield_now();
        return ProbeImportReturn::integer(0);
    }
    if is_guest_sync_import(nid) {
        let is_mutex_init = matches!(nid, "cmo1RIYva9o" | "ttHNfU+qDBU");
        let mutex_attributes = if is_mutex_init {
            let state = PROBE_DISPATCH.get_or_init(|| RwLock::new(ProbeDispatchState::default()));
            let state = state
                .read()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            pthread_mutex_attr_state(call_args[1], &state)
        } else {
            None
        };
        if is_mutex_init && call_args[1] != 0 && mutex_attributes.is_none() {
            return ProbeImportReturn::integer(sce_kernel_error(SCE_KERNEL_ERROR_INVALID_ARGUMENT));
        }
        return ProbeImportReturn::integer(dispatch_guest_sync_import(
            nid,
            call_args,
            mutex_attributes,
        ));
    }
    if matches!(
        nid,
        "5OqszGpy7Mg" | "zlfEH8FmyUA" | "VOBg+iNwB-4" | "2vDqwBlpF-o"
    ) {
        return dispatch_guest_numeric_parser(nid, call_args);
    }
    if nid == "pztV4AF18iI" {
        return ProbeImportReturn::integer(guest_sincosf(
            low_f32(float_argument0),
            call_args[0],
            call_args[1],
        ));
    }
    if nid == "AEJdIVZTEmo" {
        return ProbeImportReturn::integer(guest_qsort(call_args));
    }
    if nid == "dhK16CKwhQg" {
        return ProbeImportReturn::integer(u64::from(float_argument0.is_finite()));
    }
    let state = PROBE_DISPATCH.get_or_init(|| RwLock::new(ProbeDispatchState::default()));
    let mut state = state
        .write()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    if nid == "H2e8t5ScQGc" {
        let callbacks = take_exit_callbacks(&mut state, Some(call_args[0]));
        drop(state);
        invoke_guest_exit_callbacks(callbacks);
        return ProbeImportReturn::integer(0);
    }
    if nid == "6UgtwV+0zb4" {
        if call_args[0] == 0 || call_args[2] == 0 {
            return ProbeImportReturn::integer(sce_kernel_error(SCE_KERNEL_ERROR_INVALID_ARGUMENT));
        }
        let attributes = pthread_attr_state(call_args[1], &state).unwrap_or_default();
        let entry = call_args[2];
        let thread_argument = call_args[3];
        let guest_tls = clone_guest_tls(&state.tls_template);
        let thread_name = read_guest_cstring(call_args[4]).filter(|name| !name.is_empty());
        let Some(handle) =
            allocate_pthread_thread_object(&mut state, thread_name.as_deref().unwrap_or_default())
        else {
            return ProbeImportReturn::integer(sce_kernel_error(SCE_KERNEL_ERROR_TRY_AGAIN));
        };
        let mut builder = std::thread::Builder::new();
        if let Ok(stack_size) = usize::try_from(attributes.stack_size)
            && stack_size >= usize::try_from(MACOS_GUEST_PAGE_ALIGNMENT).unwrap_or(0x4000)
        {
            builder = builder.stack_size(stack_size);
        }
        if let Some(name) = thread_name.as_ref() {
            builder = builder.name(name.clone());
        }
        if trace_guest_sync() {
            eprintln!(
                "[REMu][THREAD] create handle={handle} name={:?} entry={entry:#x} arg={thread_argument:#x} detach={} stack={:#x}",
                thread_name.as_deref().unwrap_or(""),
                attributes.detach_state,
                attributes.stack_size
            );
        }
        let trace_name = thread_name.clone();
        let Ok(thread) = builder.spawn(move || {
            if trace_guest_sync() {
                eprintln!(
                    "[REMu][THREAD] start handle={handle} name={:?}",
                    trace_name.as_deref().unwrap_or("")
                );
            }
            set_current_guest_thread_handle(handle);
            set_current_guest_thread_pointer(guest_tls.thread_pointer);
            let result = run_guest_thread(entry, thread_argument);
            set_current_guest_thread_pointer(0);
            set_current_guest_thread_handle(0);
            drop(guest_tls);
            if trace_guest_sync() {
                eprintln!(
                    "[REMu][THREAD] exit handle={handle} name={:?} result={result:#x}",
                    trace_name.as_deref().unwrap_or("")
                );
            }
            retire_detached_guest_thread(handle);
            result
        }) else {
            if let Some(object) = state.pthread_threads.remove(&handle) {
                guest_free(object.handle);
            }
            return ProbeImportReturn::integer(sce_kernel_error(SCE_KERNEL_ERROR_TRY_AGAIN));
        };
        write_guest_u64(call_args[0], handle);
        state.thread_attributes.insert(handle, attributes);
        if attributes.detach_state == 0 {
            state.thread_handles.insert(handle, thread);
        } else {
            state.detached_threads.insert(handle);
            drop(thread);
        }
        return ProbeImportReturn::integer(0);
    }
    if nid == "onNY9Byn-W8" {
        if call_args[0] == 0 || state.detached_threads.contains(&call_args[0]) {
            return ProbeImportReturn::integer(sce_kernel_error(SCE_KERNEL_ERROR_INVALID_ARGUMENT));
        }
        let thread_id = call_args[0];
        let handle = state.thread_handles.remove(&thread_id);
        let Some(handle) = handle else {
            return ProbeImportReturn::integer(sce_kernel_error(SCE_KERNEL_ERROR_NOT_FOUND));
        };
        let object = state.pthread_threads.remove(&thread_id);
        state.thread_attributes.remove(&call_args[0]);
        drop(state);
        let return_value = handle.join().unwrap_or(u64::MAX);
        if let Some(object) = object {
            guest_free(object.handle);
        }
        if call_args[1] != 0 {
            write_guest_u64(call_args[1], return_value);
        }
        return ProbeImportReturn::integer(0);
    }
    if nid == "4qGrR6eoP9Y" {
        let handle = call_args[0];
        if handle == 0 || state.detached_threads.contains(&handle) {
            return ProbeImportReturn::integer(sce_kernel_error(SCE_KERNEL_ERROR_INVALID_ARGUMENT));
        }
        let Some(thread) = state.thread_handles.remove(&handle) else {
            return ProbeImportReturn::integer(sce_kernel_error(SCE_KERNEL_ERROR_NOT_FOUND));
        };
        let finished = thread.is_finished();
        state.detached_threads.insert(handle);
        if let Some(attributes) = state.thread_attributes.get_mut(&handle) {
            attributes.detach_state = 1;
        }
        drop(thread);
        if finished {
            state.detached_threads.remove(&handle);
            state.thread_attributes.remove(&handle);
            if let Some(object) = state.pthread_threads.remove(&handle) {
                guest_free(object.handle);
            }
        }
        return ProbeImportReturn::integer(0);
    }
    if nid == "QOQtbeDqsT4" {
        let output = state.audio_out.output(call_args);
        drop(state);
        return match output {
            Ok(delay) => {
                if !delay.is_zero() {
                    std::thread::sleep(delay);
                }
                ProbeImportReturn::integer(0)
            }
            Err(error) => ProbeImportReturn::integer(audio_out_error(error)),
        };
    }
    ProbeImportReturn::integer(dispatch_probe_import(
        nid, call_args, stack_args, &mut state,
    ))
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
struct ProbeImportReturn {
    integer: u64,
    float: f64,
}

impl ProbeImportReturn {
    const fn integer(integer: u64) -> Self {
        Self {
            integer,
            float: 0.0,
        }
    }

    const fn float(float: f64) -> Self {
        Self { integer: 0, float }
    }
}

struct GuestThreadTls {
    _storage: Vec<u8>,
    thread_pointer: u64,
}

fn clone_guest_tls(template: &[u8]) -> GuestThreadTls {
    let size = usize::try_from(GUEST_TLS_PREFIX_SIZE + GUEST_TLS_TCB_SIZE)
        .expect("guest TLS allocation size fits usize");
    assert_eq!(
        template.len(),
        size,
        "guest TLS template has the linked size"
    );
    let mut storage = template.to_vec();
    let thread_pointer_offset =
        usize::try_from(GUEST_TLS_PREFIX_SIZE).expect("guest TLS prefix size fits usize");
    let original_thread_pointer = u64::from_le_bytes(
        storage[thread_pointer_offset..thread_pointer_offset + 8]
            .try_into()
            .expect("thread pointer word is present in TLS template"),
    );
    let thread_pointer = storage.as_mut_ptr() as usize as u64 + GUEST_TLS_PREFIX_SIZE;
    let dtv_offset = thread_pointer_offset + 0x100;
    let maximum_module_id = u64::from_le_bytes(
        storage[dtv_offset + 8..dtv_offset + 16]
            .try_into()
            .expect("DTV module count is present in TLS template"),
    );
    for module_id in 1..=maximum_module_id {
        let Ok(entry_offset) = usize::try_from(module_id.saturating_mul(16)) else {
            continue;
        };
        let entry_offset = dtv_offset + entry_offset;
        let original_block = u64::from_le_bytes(
            storage[entry_offset..entry_offset + 8]
                .try_into()
                .expect("DTV module entry is present in TLS template"),
        );
        if original_block == 0 {
            continue;
        }
        let displacement = i128::from(original_block) - i128::from(original_thread_pointer);
        let rebased = i128::from(thread_pointer) + displacement;
        if let Ok(rebased) = u64::try_from(rebased) {
            storage[entry_offset..entry_offset + 8].copy_from_slice(&rebased.to_le_bytes());
        }
    }
    write_guest_u64(thread_pointer, thread_pointer);
    write_guest_u64(thread_pointer + 0x08, thread_pointer + 0x100);
    write_guest_u64(thread_pointer + 0x10, thread_pointer);
    write_guest_u64(thread_pointer + 0x60, thread_pointer);
    write_guest_u64(thread_pointer + 0x100, 1);
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
            | "mkx2fVhNMsg"
            | "188x57JYp0g"
            | "Zxa0VhQVTsk"
            | "4czppHBiriw"
    )
}

fn dispatch_guest_sync_import(
    nid: &str,
    arguments: [u64; 6],
    mutex_attributes: Option<PthreadMutexAttrState>,
) -> u64 {
    match nid {
        "cmo1RIYva9o" | "ttHNfU+qDBU" => {
            guest_mutex_init(arguments[0], mutex_attributes.unwrap_or_default())
        }
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
        "kDh-NfxgMtE" | "JGgj7Uvrl+A" | "mkx2fVhNMsg" => guest_condition_signal(arguments[0]),
        "188x57JYp0g" => guest_semaphore_create(arguments),
        "Zxa0VhQVTsk" => guest_semaphore_wait(arguments),
        "4czppHBiriw" => guest_semaphore_signal(arguments),
        _ => 0,
    }
}

fn guest_sync() -> &'static (Mutex<GuestSyncState>, Condvar) {
    GUEST_SYNC.get_or_init(|| (Mutex::new(GuestSyncState::default()), Condvar::new()))
}

fn guest_mutex_init(address: u64, attributes: PthreadMutexAttrState) -> u64 {
    if address == 0 {
        return sce_kernel_error(SCE_KERNEL_ERROR_INVALID_ARGUMENT);
    }
    let sync = guest_sync();
    let mut state = sync
        .0
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    if state.mutexes.contains_key(&address) {
        return sce_kernel_error(SCE_KERNEL_ERROR_INVALID_ARGUMENT);
    }
    state.mutexes.insert(
        address,
        GuestMutexState {
            mutex_type: attributes.mutex_type,
            ..GuestMutexState::default()
        },
    );
    0
}

fn guest_mutex_destroy(address: u64) -> u64 {
    if address == 0 {
        return sce_kernel_error(SCE_KERNEL_ERROR_INVALID_ARGUMENT);
    }
    let sync = guest_sync();
    let mut state = sync
        .0
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    match state.mutexes.get(&address) {
        Some(mutex) if mutex.owner.is_some() => u64::from(libc::EBUSY.cast_unsigned()),
        Some(_) => {
            state.mutexes.remove(&address);
            0
        }
        None => sce_kernel_error(SCE_KERNEL_ERROR_NOT_FOUND),
    }
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
                return match mutex.mutex_type {
                    2 | 3 => {
                        mutex.recursion = mutex.recursion.saturating_add(1);
                        0
                    }
                    4 if !try_only => 0,
                    _ if try_only => u64::from(libc::EBUSY.cast_unsigned()),
                    _ => u64::from(libc::EDEADLK.cast_unsigned()),
                };
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
    if trace_guest_sync() {
        eprintln!(
            "[REMu][SEMA] create handle={handle} name={:?} initial={initial} maximum={maximum}",
            read_guest_cstring(arguments[1]).unwrap_or_default()
        );
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
            if trace_guest_sync() {
                eprintln!(
                    "[REMu][SEMA] acquire handle={handle} needed={needed} remaining={}",
                    semaphore.count
                );
            }
            return 0;
        }
        if trace_guest_sync() {
            eprintln!(
                "[REMu][SEMA] wait handle={handle} needed={needed} available={}",
                semaphore.count
            );
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
    if trace_guest_sync() {
        eprintln!(
            "[REMu][SEMA] signal handle={handle} amount={amount} available={}",
            semaphore.count
        );
    }
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

fn low_f32(value: f64) -> f32 {
    let bytes = value.to_bits().to_le_bytes();
    f32::from_bits(u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]))
}

fn guest_atomic_fetch_4(address: u64, value: u64, subtract: bool) -> u64 {
    if address == 0 || !address.is_multiple_of(std::mem::align_of::<AtomicU32>() as u64) {
        return u64::MAX;
    }
    let Ok(host_address) = usize::try_from(address) else {
        return u64::MAX;
    };
    let pointer = host_address as *mut u32;
    // SAFETY: Guest atomic imports provide a live, naturally aligned four-byte object in the
    // shared process address space. SeqCst is at least as strong as every guest memory-order mode.
    let atomic = unsafe { AtomicU32::from_ptr(pointer) };
    let value = low_u32(value);
    let previous = if subtract {
        atomic.fetch_sub(value, Ordering::SeqCst)
    } else {
        atomic.fetch_add(value, Ordering::SeqCst)
    };
    u64::from(previous)
}

fn dispatch_kernel_file_import(
    nid: &str,
    arguments: [u64; 6],
    state: &mut ProbeDispatchState,
) -> Option<u64> {
    // These raw libKernel calls use Gen5 kernel errors, not POSIX -1/errno semantics.
    Some(match nid {
        "1G3lF1Gg1k8" => guest_kernel_open(arguments, state),
        "eV9wAD2riIA" => guest_kernel_stat(arguments, state),
        "Cg4srZ6TKbU" => guest_kernel_read(arguments, state),
        "UK2Tl2DWUns" => guest_kernel_close(arguments, state),
        "oib76F-12fk" => guest_kernel_lseek(arguments, state),
        "kBwCPsYX-m4" => guest_kernel_fstat(arguments, state),
        "VW3TVZiM4-E" => guest_kernel_ftruncate(arguments, state),
        _ => return None,
    })
}

#[allow(clippy::match_same_arms, clippy::too_many_lines)]
fn dispatch_probe_import(
    nid: &str,
    arguments: [u64; 6],
    stack_arguments: [u64; 6],
    state: &mut ProbeDispatchState,
) -> u64 {
    if let Some(result) = dispatch_kernel_file_import(nid, arguments, state) {
        return result;
    }
    if let Some(result) = agc_command_abi::dispatch_agc_command_abi(nid, arguments, stack_arguments)
    {
        return result;
    }
    match nid {
        // Process-runtime initialization and destructor registration are observable lifecycle
        // operations. Both C and Itanium callbacks are retained in one registration-order list.
        "bzQExy189ZI" => initialize_guest_libc(arguments[0], state),
        "8G2LB+A3rzg" => register_guest_exit_callback(arguments[0], None, 0, state),
        "tsvEmnenz48" => {
            register_guest_exit_callback(arguments[0], Some(arguments[1]), arguments[2], state)
        }
        // The game-bundled C runtime registers its per-thread cleanup hooks during DT_INIT and
        // reads the main executable's PT_SCE_PROCPARAM block to configure its allocator.
        "rNhWz+lvOMU" => {
            state.thread_dtors_callback = arguments[0];
            0
        }
        "pB-yGZ2nQ9o" => {
            state.thread_atexit_count_callback = arguments[0];
            0
        }
        "WhCc1w3EhSI" => {
            state.thread_atexit_report_callback = arguments[0];
            0
        }
        "959qrazPIrg" => state.process_param_address,
        "p5EcQeEeJAE" => {
            state.application_heap_api_address = arguments[0];
            0
        }
        "bnZxYgAFeA0" => sanitizer_new_replace_external(),
        "NWtTN10cJzE" => libc_heap_get_trace_info(arguments[0], state),
        "6c3rCVE-fTU" | "wuCroIGjt2g" => {
            guest_posix_file_result(guest_kernel_open(arguments, state))
        }
        "DRuBt2pvICk" | "AqBioC2vF3I" => {
            guest_posix_file_result(guest_kernel_read(arguments, state))
        }
        "NNtFaKJbPt0" | "bY-PO6JhzhQ" => {
            guest_posix_file_result(guest_kernel_close(arguments, state))
        }
        "FxVZqBAA7ks" | "FN4gaPmuFV8" => {
            guest_posix_file_result(guest_kernel_write(arguments, state))
        }
        "Oy6IpwgtYOk" => guest_posix_file_result(guest_kernel_lseek(arguments, state)),
        "E6ao34wPw+U" => guest_posix_file_result(guest_kernel_stat(arguments, state)),
        "mqQMh1zPPT8" => guest_posix_file_result(guest_kernel_fstat(arguments, state)),
        "ih4CD9-gghM" => guest_posix_file_result(guest_kernel_ftruncate(arguments, state)),
        "4wSze92BhLI" => guest_kernel_write(arguments, state),
        "aI+OeCz8xrQ" => current_guest_thread_handle(),
        // sceKernelGetDirectMemorySize
        "pO96TwzOm5E" => 16 * 1024 * 1024 * 1024,
        // sceKernelAllocateDirectMemory
        "rTXw65xmLIA" => allocate_direct_memory(arguments, state),
        // sceKernelMapDirectMemory
        "L-Q3LEjIbgA" => map_direct_memory(arguments, state),
        // sceKernelMapDirectMemory2 inserts memory type and passes alignment as argument seven.
        "BQQniolj9tQ" => map_direct_memory_2(arguments, stack_arguments[0], state),
        // Kernel VM ownership calls must describe and retire the mappings created above. Returning
        // success without filling VirtualQuery makes libc feed uninitialized lengths into munmap.
        "rVjRvHJ0X6c" => kernel_virtual_query(arguments, state),
        "cQke9UuBQOk" => kernel_munmap(arguments, state),
        "hwVSPCmp5tM" => kernel_checked_release_direct_memory(arguments, state),
        "MBuItvba6z8" => kernel_release_direct_memory(arguments, state),
        // libc memset
        "8zTFvBIAIN8" => guest_memset(arguments[0], arguments[1], arguments[2]),
        // libc memcpy
        "Q3VBxCXhUHs" => guest_memcpy(arguments[0], arguments[1], arguments[2]),
        // libc memmove
        "+P6FRGH4LfA" => guest_memmove(arguments[0], arguments[1], arguments[2]),
        // libc memcmp
        "DfivPArhucg" => guest_memcmp(arguments[0], arguments[1], arguments[2]),
        // libc memchr
        "8u8lPzUEq+U" => guest_memchr(arguments[0], arguments[1], arguments[2]),
        // Gen5 uses two-byte wchar_t values. macOS uses four-byte wchar_t values, so these
        // functions must not be bound directly to the host libc wide-memory family.
        "fL3O02ypZFE" => guest_wmemcpy(arguments[0], arguments[1], arguments[2]),
        "Noj9PsJrsa8" => guest_wmemmove(arguments[0], arguments[1], arguments[2]),
        "fnUEjBCNRVU" => guest_wmemchr(arguments[0], arguments[1], arguments[2]),
        "QJ5xVfKkni0" => guest_wmemcmp(arguments[0], arguments[1], arguments[2]),
        // The guest ABI returns the previous four-byte value; the third argument is memory order.
        "2HnmKiLmV6s" => guest_atomic_fetch_4(arguments[0], arguments[1], true),
        "iPBqs+YUUFw" => guest_atomic_fetch_4(arguments[0], arguments[1], false),
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
        "viiwFMaNamA" => guest_strstr(arguments[0], arguments[1]),
        // libc stdio uses host FILE objects after resolving console paths inside the game dump.
        "xeYO4u7uyJ0" => guest_fopen(arguments[0], arguments[1], state),
        "rQFVBXp-Cxg" => guest_fseek(arguments[0], arguments[1], arguments[2]),
        "Qazy8LmXTvw" => guest_ftell(arguments[0]),
        "uodLYyUip20" => guest_fclose(arguments[0]),
        "lbB+UlZqVG0" => guest_fread(arguments[0], arguments[1], arguments[2], arguments[3]),
        // Dinkumware's locale helpers are used by the save-error formatting path. The character
        // table accessor must return a live table; its lock and lifetime helpers are process-local
        // bookkeeping in the single-process HLE runtime.
        "sUP1hBaouOw" => guest_ctype_table(state),
        "1uJgoVq3bQU" => guest_tolower_table(state),
        "rcQCUr0EaRU" => guest_toupper_table(state),
        "9rMML086SEE" | "hEQ2Yi4PJXA" => guest_global_locale(state),
        "kALvdgEv5ME" | "9nf8joUTSaQ" | "Q1BL70XVV0o" => 0,
        "hqi8yMOCmG0" | "p6LrHjIQMdk" | "QW2jL1J5rwY" | "P8F2oavZXtY" => arguments[0],
        // libc time returns Unix seconds and optionally stores the same value through its pointer.
        "wLlFkwG9UcQ" => guest_time(arguments[0]),
        // Preserve the SysV variadic register layout for the integer/pointer formats used by the
        // game's asset path builder. Floating-point varargs require a wider import frame later.
        "tcVi5SivF7Q" => guest_sprintf(arguments),
        // libc fputs
        "QrZZdJ8XsX0" => guest_fputs(arguments[0]),
        // Itanium C++ one-time initialization guards.
        "3GPpjQdAMTw" => guest_guard_acquire(arguments[0]),
        "9rAeANT2tyE" => guest_guard_release(arguments[0]),
        "2emaaluWzUw" => guest_guard_abort(arguments[0]),
        // libc++abi exception storage. Throwing currently ends the probe with a decoded diagnostic
        // until guest unwind metadata is registered with the host unwinder.
        "cfAXurvfl5o" => guest_cxa_allocate_exception(arguments[0]),
        "nOIEswYD4Ig" => guest_cxa_free_exception(arguments[0]),
        "vkuuLfhnSZI" => guest_cxa_throw(arguments),
        // libc __error returns a stable, writable errno cell owned by the current guest thread.
        "9BcDykPmo1I" => guest_errno_address(),
        // libKernel pthread attributes are opaque allocated objects. Their scheduling, stack, and
        // detach values feed thread creation and remain observable through the matching getters.
        "F8bUHwAG284" => create_pthread_mutex_attr(arguments[0], state),
        "smWEktiyyG0" => destroy_pthread_mutex_attr(arguments[0], state),
        "iMp8QpE+XO4" => pthread_mutex_attr_set_type(arguments, state),
        "nsYoNRywwNg" => create_pthread_attr(arguments[0], PthreadAttrState::default(), state),
        "62KCwEMmzcM" => destroy_pthread_attr(arguments[0], state),
        "-Wreprtu0Qs" => pthread_attr_set_detach_state(arguments, state),
        "UTXzJbWhhTE" => pthread_attr_set_stack_size(arguments, state),
        "DzES9hQF4f4" => pthread_attr_set_schedule_param(arguments, state),
        "x1X76arYMxU" => pthread_attr_get(arguments, state),
        "-quPa4SEJUw" => pthread_attr_get_stack(arguments, state),
        "8+s5BzZjxSg" => pthread_attr_get_affinity(arguments, state),
        "m5-2bsNfv7s" => pthread_condition_attr_init(arguments[0], state),
        "waPcxYiR3WA" => pthread_condition_attr_destroy(arguments[0], state),
        // Initial single-device handles for input, video, and audio.
        "xk0AcarP3V4" => pad_open(arguments, state),
        "Up36PTk687E" => video_out_open(arguments, state),
        // libSceAudioOut owns validated ports and consumes complete configured PCM buffers through
        // the host mixer. Output itself is handled above so pacing never holds the import lock.
        "JfEPXVxhFqA" => {
            state.audio_out.initialize();
            0
        }
        "ekNvsT22rsY" => state
            .audio_out
            .open(arguments)
            .unwrap_or_else(audio_out_error),
        "s1--uE9mBFw" => state
            .audio_out
            .close(arguments[0])
            .map_or_else(audio_out_error, |()| 0),
        "b+uAV89IlxE" => state
            .audio_out
            .set_volume(arguments)
            .map_or_else(audio_out_error, |()| 0),
        "GrQ9s4IrNaQ" => state
            .audio_out
            .get_port_state(arguments)
            .map_or_else(audio_out_error, |()| 0),
        // AGC shader construction and VideoOut mutate caller-owned descriptors. Games consume the
        // resulting pointers and command data immediately, so successful no-ops are insufficient.
        "f3dg2CSgRKY" => agc_create_shader(arguments, state),
        "TRO721eVt4g" => agc_dcb_reset_queue(arguments),
        "ZvwO9euwYzc" => agc_dcb_set_registers_indirect(arguments, AGC_R_CX_REGS_INDIRECT),
        "-HOOCn0JY48" => agc_dcb_set_registers_indirect(arguments, AGC_R_SH_REGS_INDIRECT),
        "hvUfkUIQcOE" => agc_dcb_set_registers_indirect(arguments, AGC_R_UC_REGS_INDIRECT),
        "vcmNN+AAXnY" | "Qrj4c+61z4A" | "6lNcCp+fxi4" => agc_set_indirect_patch_address(arguments),
        "d-6uF9sZDIU" | "z2duB-hHQSM" | "vRoArM9zaIk" => {
            agc_add_indirect_patch_registers(arguments)
        }
        "D9sr1xGUriE" => agc_create_primitive_state(arguments),
        "HV4j+E0MBHE" => agc_create_interpolant_mapping(arguments),
        "n2fD4A+pb+g" => agc_cb_set_sh_register_range_direct(arguments),
        "V++UgBtQhn0" => agc_get_data_packet_payload_address(arguments),
        "aJf+j5yntiU" => agc_dcb_event_write(arguments),
        "LtTouSCZjHM" => agc_cb_nop(arguments),
        "WmAc2MEj6Io" => agc_dcb_dma_data(arguments, stack_arguments),
        "wr23dPKyWc0" => agc_cb_release_mem(arguments, stack_arguments),
        "VmW0Tdpy420" => agc_dcb_wait_reg_mem(arguments, stack_arguments),
        "IxYiarKlXxM" => agc_dma_data_patch_destination(arguments),
        "3KDcnM3lrcU" => agc_wait_reg_mem_patch_address(arguments),
        "0fWWK5uG9rQ" => agc_release_mem_patch_address(arguments),
        "57labkp+rSQ" => agc_dcb_acquire_mem(arguments, stack_arguments),
        "l4fM9K-Lyks" => agc_dcb_set_index_buffer(arguments, state),
        "GIIW2J37e70" => agc_dcb_set_index_size(arguments),
        "B+aG9DUnTKA" => agc_dcb_draw_index_offset(arguments),
        "tSBxhAPyytQ" => agc_dcb_set_num_instances(arguments),
        "Yw0jKSqop+E" => agc_dcb_draw_index_auto(arguments),
        "YUeqkyT7mEQ" => agc_dcb_set_flip(arguments),
        "UglJIZjGssM" => agc_driver_submit_dcb(arguments, state),
        "6UzEidRZwkg" => agc_driver_submit_multi_dcbs(arguments, state),
        "gSRnr79F8tQ" => agc_driver_submit_acb(arguments, state),
        "HF3YllT3mXU" => agc_driver_submit_multi_acbs(arguments, state),
        "PjS5uASwcV8" => video_out_set_buffer_attribute2(arguments, state),
        "rKBUtgRrtbk" => video_out_register_buffers2(arguments, state),
        "HXzjK9yI30k" => video_out_add_flip_event(arguments, state),
        // libScePad output structures must always be initialized; successful no-ops would expose
        // stale guest stack contents as controller capabilities and button state.
        "clVvL4ZDntw" => pad_set_motion_sensor_state(arguments, state),
        "gjP9-KQzoUk" => pad_get_controller_information(arguments, state),
        "YndgXqQVV7c" => pad_read_state(arguments, state),
        "hv1luiJrqQM" => pad_initialize(state),
        // libSceSystemService status used by the foreground/background state machine.
        "rPo6tV8D9bM" => system_service_get_status(arguments),
        "fZo48un7LK4" => system_service_param_get_int(arguments),
        "Vo5V8KAwCmk" => system_service_hide_splash(state),
        // libSceSaveData uses durable host-side transactions while preserving the guest ABI.
        "TywrFKCoLGY" => save_data_initialize(state),
        "dyIhnXq-0SM" => save_data_dir_name_search(arguments, state),
        "ZP4e7rlzOUk" => save_data_mount3(arguments, state),
        "gjRZNnw0JPE" => save_data_create_transaction_resource(arguments, state),
        "uW4vfTwMQVo" => save_data_umount2(arguments, state),
        "sDCBrmc61XU" => save_data_prepare(arguments, state),
        "85zul--eGXs" => save_data_set_param(arguments, state),
        "c88Yy54Mx0w" => save_data_save_icon(arguments, state),
        "ie7qhZ4X0Cc" => save_data_commit(arguments, state),
        // User-service IDs use the retail slot encoding expected by the game's controller layer.
        "CdWp0oHWGr0" => write_output_i32(arguments[0], 0x1000_0000),
        "fPhymKNvK-A" => user_service_get_login_user_id_list(arguments),
        "j3YMu1MVNNo" => user_service_initialize(state),
        // Runtime services retain initialization and loaded-module state instead of accepting
        // unrelated calls through the unknown-import fallback.
        "g8cM39EUZ6o" => sysmodule_load(arguments[0], state),
        "hZIg1EWGsHM" => ult_initialize(arguments[0], state),
        "uoUpLGNkygk" => common_dialog_initialize(state),
        "ERKzksauAJA" => state
            .services
            .save_data_dialog_status
            .cast_unsigned()
            .into(),
        // sceKernelCreateEqueue
        "D0OdFMjp46I" => write_output_u64(arguments[0], 1),
        // sceKernelWaitEqueue. Until GPU completion events are connected, report an empty wait
        // deterministically. Not writing outCount made the game interpret old stack bytes as user
        // login events and overflow its four-controller table.
        "fzyMKs9kim0" => kernel_wait_equeue(arguments, state),
        // Stateful offline NP services. Context, handle, event/property, intent and callback
        // ownership follow the guest ABI even though REmu deliberately does not upload telemetry
        // or connect a dumped title to PlayStation Network.
        "sjaobBgqeB4" => np_universal_data_initialize(arguments[0], state),
        "47UAEuQl+iI" => np_universal_data_terminate(state),
        "5zBnau1uIEo" => np_universal_data_create_context(arguments, state),
        "wB7IWzGp2v0" => np_universal_data_destroy_context(arguments[0], state),
        "hT0IAEvN+M0" => np_universal_data_create_handle(arguments, state),
        "jZCqWFgMehE" => np_universal_data_abort_handle(arguments[0], state),
        "AUIHb7jUX3I" => np_universal_data_destroy_handle(arguments[0], state),
        "tpFJ8LIKvPw" => np_universal_data_register_context(arguments, state),
        "p+GcLqwpL9M" => np_universal_data_create_event(arguments, state),
        "wG+84pnNIuo" => np_universal_data_destroy_event(arguments[0], state),
        "YE4dbtbz6OE" => np_universal_data_object_set_i32(arguments, state),
        "MfDb+4Nln64" => np_universal_data_object_set_string(arguments, state),
        "Wxbg5x3pTXA" => np_universal_data_object_set_array(arguments, state),
        "4llLk7YJRTE" => np_universal_data_array_set_string(arguments, state),
        "CzkKf7ahIyU" => np_universal_data_post_event(arguments, state),
        "m87BHxt-H60" => np_game_intent_initialize(arguments[0], state),
        "jEIXUAr9XE8" => np_game_intent_receive(arguments[0], state),
        "rPl0INNc-M8" => np_game_intent_get_property_string(arguments, state),
        "Bagshr7OQ6Q" => np_trophy_create_context(arguments, state),
        "Gz1rmUZpROM" => np_trophy_create_handle(arguments, state),
        "sysY2FHYff4" => np_trophy_destroy_context(arguments[0], state),
        "d8P11CI40KE" => np_trophy_destroy_handle(arguments[0], state),
        "fYapWA9xVmA" => np_trophy_abort_handle(arguments[0], state),
        "bIDov3wBu5Q" => np_trophy_register_context(arguments, state),
        "sUXGfNMalIo" => np_trophy_register_unlock_callback(arguments, state),
        "wVqxM58sIKs" => np_trophy_unregister_unlock_callback(state),
        // Exact Gen5 register-default descriptors consumed by indirect AGC state packets.
        "2JtWUUiYBXs" => agc_register_defaults(arguments[0], false, state),
        "wRbq6ZjNop4" => agc_register_defaults(arguments[0], true, state),
        "23LRUSvYu1M" => agc_initialize(arguments, state),
        "h9z6+0hEydk" => {
            std::thread::yield_now();
            0
        }
        "vSMAm3cxYTY" => kernel_mprotect(arguments, state),
        // sceKernelUsleep
        "1jfXLRVzisc" => guest_usleep(arguments[0]),
        // POSIX clock_gettime. Darwin's timespec layout matches the x86-64 guest ABI.
        "lLMT9vJAck0" => guest_clock_gettime(arguments[0], arguments[1]),
        // Keep discovery moving, but identify the exact calls that reached the fallback. The
        // general first-hit log also includes known imports whose implementation lives above this
        // dispatcher, so it cannot distinguish a missing ABI from an incomplete name table.
        _ => {
            if state.unimplemented_nids.insert(nid.to_owned()) {
                eprintln!(
                    "[REMu][IMPORT] unimplemented nid={nid} args={:#x},{:#x},{:#x},{:#x},{:#x},{:#x}",
                    arguments[0],
                    arguments[1],
                    arguments[2],
                    arguments[3],
                    arguments[4],
                    arguments[5]
                );
            }
            sce_kernel_error(SCE_KERNEL_ERROR_NOT_IMPLEMENTED)
        }
    }
}

fn kernel_wait_equeue(arguments: [u64; 6], state: &mut ProbeDispatchState) -> u64 {
    let handle = arguments[0];
    let events = arguments[1];
    let event_capacity = low_u32(arguments[2]);
    let out_count = arguments[3];
    if events == 0 || event_capacity == 0 {
        return u64::from(libc::EINVAL.cast_unsigned());
    }
    if let Some(index) = state
        .pending_events
        .iter()
        .position(|event| event.equeue == handle)
    {
        let event = state
            .pending_events
            .remove(index)
            .expect("the queued event index was just found");
        write_guest_u64(events, event.ident);
        write_guest_u16(events + 0x08, event.filter.cast_unsigned());
        write_guest_u16(events + 0x0a, event.flags);
        write_guest_u32(events + 0x0c, event.fflags);
        write_guest_u64(events + 0x10, event.data);
        write_guest_u64(events + 0x18, event.user_data);
        if out_count != 0 {
            write_guest_u32(out_count, 1);
        }
        return 0;
    }
    if out_count != 0 {
        write_guest_u32(out_count, 0);
    }
    // Avoid letting the temporary empty-queue implementation turn the render worker into a hot
    // spin. A real wait will be woken by the Metal submission/completion path.
    std::thread::sleep(Duration::from_micros(100));
    0
}

fn initialize_guest_libc(environment: u64, state: &mut ProbeDispatchState) -> u64 {
    match state.services.libc_environment {
        Some(previous) if previous != environment => {
            sce_kernel_error(SCE_KERNEL_ERROR_INVALID_ARGUMENT)
        }
        Some(_) => 0,
        None => {
            state.services.libc_environment = Some(environment);
            0
        }
    }
}

fn register_guest_exit_callback(
    function: u64,
    argument: Option<u64>,
    module: u64,
    state: &mut ProbeDispatchState,
) -> u64 {
    if function == 0 {
        return u64::MAX;
    }
    state.exit_callbacks.push(GuestExitCallback {
        function,
        argument,
        module,
    });
    0
}

fn take_exit_callbacks(
    state: &mut ProbeDispatchState,
    module: Option<u64>,
) -> Vec<GuestExitCallback> {
    let mut callbacks = Vec::new();
    let mut index = state.exit_callbacks.len();
    while index != 0 {
        index -= 1;
        let callback = state.exit_callbacks[index];
        let matches = module.is_none_or(|module| {
            module == 0 || (callback.argument.is_some() && callback.module == module)
        });
        if matches {
            callbacks.push(state.exit_callbacks.remove(index));
        }
    }
    callbacks
}

fn invoke_guest_exit_callbacks(callbacks: Vec<GuestExitCallback>) {
    for callback in callbacks {
        if let Some(argument) = callback.argument {
            type Destructor = unsafe extern "C" fn(u64);
            // SAFETY: The guest registered a live Itanium destructor function and its opaque
            // argument. It remains executable until the corresponding module is finalized.
            let destructor: Destructor = unsafe { std::mem::transmute(callback.function) };
            // SAFETY: The registration defines the function and argument ABI above.
            unsafe { destructor(argument) };
        } else {
            type Callback = unsafe extern "C" fn();
            // SAFETY: C atexit registers a live no-argument callback in guest executable memory.
            let callback: Callback = unsafe { std::mem::transmute(callback.function) };
            // SAFETY: The registration defines the no-argument callback ABI above.
            unsafe { callback() };
        }
    }
}

fn run_all_guest_exit_callbacks() {
    loop {
        let callbacks = PROBE_DISPATCH.get().map_or_else(Vec::new, |state| {
            let mut state = state
                .write()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            take_exit_callbacks(&mut state, None)
        });
        if callbacks.is_empty() {
            return;
        }
        invoke_guest_exit_callbacks(callbacks);
    }
}

fn sysmodule_load(module: u64, state: &mut ProbeDispatchState) -> u64 {
    let module = low_i32(module);
    if module < 0 {
        return sce_kernel_error(SCE_KERNEL_ERROR_INVALID_ARGUMENT);
    }
    state.services.loaded_sysmodules.insert(module);
    0
}

fn ult_initialize(runtime: u64, state: &mut ProbeDispatchState) -> u64 {
    if runtime == 0 {
        return sce_kernel_error(SCE_KERNEL_ERROR_INVALID_ARGUMENT);
    }
    match state.services.ult_runtime {
        Some(previous) if previous != runtime => {
            sce_kernel_error(SCE_KERNEL_ERROR_INVALID_ARGUMENT)
        }
        Some(_) => 0,
        None => {
            state.services.ult_runtime = Some(runtime);
            0
        }
    }
}

fn system_service_hide_splash(state: &mut ProbeDispatchState) -> u64 {
    state.services.splash = SplashState::Hidden;
    0
}

fn user_service_initialize(state: &mut ProbeDispatchState) -> u64 {
    state.services.user_service = InitializationState::Initialized;
    0
}

fn common_dialog_initialize(state: &mut ProbeDispatchState) -> u64 {
    state.services.common_dialog = InitializationState::Initialized;
    0
}

fn agc_initialize(arguments: [u64; 6], state: &mut ProbeDispatchState) -> u64 {
    let agc_state = arguments[0];
    let version = low_u32(arguments[1]);
    if agc_state == 0 || !matches!(version, 7 | 8 | 10 | 13) {
        return sce_kernel_error(SCE_KERNEL_ERROR_INVALID_ARGUMENT);
    }
    state.services.agc_states.insert(agc_state, version);
    0
}

fn user_service_get_login_user_id_list(arguments: [u64; 6]) -> u64 {
    let output = arguments[0];
    if output == 0 {
        return u64::from(libc::EINVAL.cast_unsigned());
    }
    write_guest_u32(output, 0x1000_0000);
    for offset in [0x04, 0x08, 0x0c] {
        write_guest_u32(output + offset, u32::MAX);
    }
    0
}

const SCE_PAD_ERROR_INVALID_HANDLE: u32 = 0x8092_0003;
const SCE_PAD_ERROR_NOT_INITIALIZED: u32 = 0x8092_0005;
const SCE_PAD_ERROR_DEVICE_NOT_CONNECTED: u32 = 0x8092_0007;

fn pad_initialize(state: &mut ProbeDispatchState) -> u64 {
    state.services.pad = InitializationState::Initialized;
    0
}

fn pad_open(arguments: [u64; 6], state: &ProbeDispatchState) -> u64 {
    if state.services.pad != InitializationState::Initialized {
        return sce_kernel_error(SCE_PAD_ERROR_NOT_INITIALIZED);
    }
    let user = low_i32(arguments[0]);
    let port_type = low_i32(arguments[1]);
    let index = low_i32(arguments[2]);
    if user != 0x1000_0000 || port_type != 0 || index != 0 {
        return sce_kernel_error(SCE_PAD_ERROR_DEVICE_NOT_CONNECTED);
    }
    1
}

fn validate_pad_handle(handle: u64, state: &ProbeDispatchState) -> Result<(), u64> {
    if state.services.pad != InitializationState::Initialized {
        return Err(sce_kernel_error(SCE_PAD_ERROR_NOT_INITIALIZED));
    }
    if !matches!(low_i32(handle), 0 | 1) {
        return Err(sce_kernel_error(SCE_PAD_ERROR_INVALID_HANDLE));
    }
    Ok(())
}

fn pad_set_motion_sensor_state(arguments: [u64; 6], state: &ProbeDispatchState) -> u64 {
    if let Err(error) = validate_pad_handle(arguments[0], state) {
        return error;
    }
    if !matches!(low_i32(arguments[1]), 0 | 1) {
        return sce_kernel_error(SCE_KERNEL_ERROR_INVALID_ARGUMENT);
    }
    0
}

fn pad_get_controller_information(arguments: [u64; 6], state: &ProbeDispatchState) -> u64 {
    if let Err(error) = validate_pad_handle(arguments[0], state) {
        return error;
    }
    let output = arguments[1];
    if output == 0 {
        return u64::from(libc::EINVAL.cast_unsigned());
    }
    guest_memset(output, 0, 0x1c);
    write_guest_u32(output, 44.86_f32.to_bits());
    write_guest_u16(output + 0x04, 1920);
    write_guest_u16(output + 0x06, 943);
    write_guest_u8(output + 0x08, 30);
    write_guest_u8(output + 0x09, 30);
    write_guest_u8(output + 0x0b, 1);
    write_guest_u8(output + 0x0c, 1);
    0
}

fn pad_read_state(arguments: [u64; 6], state: &ProbeDispatchState) -> u64 {
    static LAST_LOGGED_BUTTONS: AtomicU32 = AtomicU32::new(u32::MAX);
    if let Err(error) = validate_pad_handle(arguments[0], state) {
        return error;
    }
    let output = arguments[1];
    if output == 0 {
        return u64::from(libc::EINVAL.cast_unsigned());
    }
    guest_memset(output, 0, 0x78);
    let host = remu_gpu::host_pad_state();
    let buttons = if auto_cross_active() {
        host.buttons | remu_gpu::PAD_BUTTON_CROSS
    } else {
        host.buttons
    };
    let previous_buttons = LAST_LOGGED_BUTTONS.swap(buttons, Ordering::Relaxed);
    if previous_buttons != buttons {
        eprintln!("[REMu][PAD] buttons={buttons:#010x}");
    }
    write_guest_u32(output, buttons);
    write_guest_u8(output + 0x04, host.left_x);
    write_guest_u8(output + 0x05, host.left_y);
    write_guest_u8(output + 0x06, host.right_x);
    write_guest_u8(output + 0x07, host.right_y);
    write_guest_u8(output + 0x08, host.l2);
    write_guest_u8(output + 0x09, host.r2);
    write_guest_u32(output + 0x18, 1.0_f32.to_bits());
    write_guest_u8(output + 0x4c, 1);
    let mut timestamp = libc::timespec {
        tv_sec: 0,
        tv_nsec: 0,
    };
    // SAFETY: timestamp is a valid writable host timespec.
    if unsafe { libc::clock_gettime(libc::CLOCK_MONOTONIC, &raw mut timestamp) } == 0 {
        let seconds = u64::try_from(timestamp.tv_sec).unwrap_or(0);
        let nanoseconds = u64::try_from(timestamp.tv_nsec).unwrap_or(0);
        write_guest_u64(
            output + 0x50,
            seconds
                .saturating_mul(1_000_000)
                .saturating_add(nanoseconds / 1_000),
        );
    }
    write_guest_u8(output + 0x68, 1);
    0
}

fn auto_cross_active() -> bool {
    static START: OnceLock<Instant> = OnceLock::new();
    static TIMES: OnceLock<Vec<Duration>> = OnceLock::new();
    static HOLD: OnceLock<Duration> = OnceLock::new();
    static PULSE: OnceLock<Mutex<(usize, Option<Instant>)>> = OnceLock::new();
    let times = TIMES.get_or_init(|| {
        std::env::var("REMU_AUTO_CROSS")
            .unwrap_or_default()
            .split(',')
            .filter_map(|value| value.trim().parse::<f64>().ok())
            .filter(|seconds| seconds.is_finite() && *seconds >= 0.0)
            .map(Duration::from_secs_f64)
            .collect()
    });
    if times.is_empty() {
        return false;
    }
    let hold = *HOLD.get_or_init(|| {
        std::env::var("REMU_AUTO_CROSS_HOLD_MS")
            .ok()
            .and_then(|value| value.parse::<u64>().ok())
            .map_or(Duration::from_millis(400), Duration::from_millis)
    });
    let start = *START.get_or_init(Instant::now);
    let now = Instant::now();
    let elapsed = now.saturating_duration_since(start);
    let mut pulse = PULSE
        .get_or_init(|| Mutex::new((0, None)))
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    if pulse.1.is_some_and(|active_until| now < active_until) {
        return true;
    }
    pulse.1 = None;
    if times
        .get(pulse.0)
        .is_some_and(|scheduled| elapsed >= *scheduled)
    {
        pulse.0 += 1;
        pulse.1 = Some(now + hold);
        return true;
    }
    false
}

fn system_service_get_status(arguments: [u64; 6]) -> u64 {
    let output = arguments[0];
    if output == 0 {
        return u64::from(libc::EINVAL.cast_unsigned());
    }
    guest_memset(output, 0, 0x0c);
    write_guest_u8(output + 0x06, 1);
    0
}

fn system_service_param_get_int(arguments: [u64; 6]) -> u64 {
    const SYSTEM_SERVICE_ERROR_PARAMETER: u32 = 0x80a1_0003;

    if arguments[1] == 0 {
        return sce_kernel_error(SYSTEM_SERVICE_ERROR_PARAMETER);
    }
    let value: i32 = match low_i32(arguments[0]) {
        1 | 2 | 3 | 1000 => 1,
        4 => 180,
        _ => 0,
    };
    write_guest_u32(arguments[1], value.cast_unsigned());
    0
}

fn agc_register_defaults(version: u64, internal: bool, state: &mut ProbeDispatchState) -> u64 {
    if !matches!(low_u32(version), 7 | 8 | 10 | 13) {
        return 0;
    }

    let cached = if internal {
        &mut state.agc_internal_register_defaults
    } else {
        &mut state.agc_primary_register_defaults
    };
    if *cached != 0 {
        return *cached;
    }

    let layout = if internal {
        agc_register_defaults::RegisterDefaultsLayout::internal()
    } else {
        agc_register_defaults::RegisterDefaultsLayout::primary()
    };
    let blob_length = layout.blob_length();
    let Ok(blob_length_u64) = u64::try_from(blob_length) else {
        return 0;
    };
    let address = guest_calloc(1, blob_length_u64);
    if address == 0 {
        return 0;
    }
    let Some(blob) = layout.build(address) else {
        guest_free(address);
        return 0;
    };
    debug_assert_eq!(blob.len(), blob_length);
    // SAFETY: `address` owns a live allocation of exactly `blob_length` bytes and the source is a
    // separate Vec allocation. The descriptor remains cached for the lifetime of the probe.
    unsafe {
        std::ptr::copy_nonoverlapping(blob.as_ptr(), address as *mut u8, blob.len());
    }
    *cached = address;
    eprintln!(
        "[REMu][AGC] register defaults internal={internal} version={} groups={} address=0x{address:016x}",
        low_u32(version),
        layout.group_count()
    );
    address
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
const AGC_SHADER_INPUT_SEMANTICS_COUNT_OFFSET: u64 = 0x50;
const AGC_SHADER_OUTPUT_SEMANTICS_COUNT_OFFSET: u64 = 0x56;

const AGC_COMMAND_BUFFER_CURSOR_UP_OFFSET: u64 = 0x10;
const AGC_COMMAND_BUFFER_CURSOR_DOWN_OFFSET: u64 = 0x18;
const AGC_COMMAND_BUFFER_RESERVED_DWORDS_OFFSET: u64 = 0x30;

const AGC_IT_NOP: u32 = 0x10;
const AGC_IT_SET_BASE: u32 = 0x11;
const AGC_IT_CLEAR_STATE: u32 = 0x12;
const AGC_IT_INDEX_BUFFER_SIZE: u32 = 0x13;
const AGC_IT_DISPATCH_DIRECT: u32 = 0x15;
const AGC_IT_DISPATCH_INDIRECT: u32 = 0x16;
const AGC_IT_INDEX_BASE: u32 = 0x26;
const AGC_IT_INDEX_TYPE: u32 = 0x2a;
const AGC_IT_DRAW_INDEX_AUTO: u32 = 0x2d;
const AGC_IT_NUM_INSTANCES: u32 = 0x2f;
const AGC_IT_DRAW_INDEX_OFFSET: u32 = 0x35;
const AGC_IT_WAIT_REG_MEM: u32 = 0x3c;
const AGC_IT_EVENT_WRITE: u32 = 0x46;
const AGC_IT_SET_SH_REG_INDIRECT: u32 = 0x63;
const AGC_IT_SET_UCONFIG_REG_INDIRECT: u32 = 0x64;
const AGC_IT_SET_CONTEXT_REG: u32 = 0x69;
const AGC_IT_SET_SH_REG: u32 = 0x76;
const AGC_IT_SET_UCONFIG_REG: u32 = 0x79;
const AGC_IT_SET_UCONFIG_REG_INDEX: u32 = 0x7a;
const AGC_IT_SET_CONTEXT_REG_INDIRECT: u32 = 0x9f;
const AGC_R_ZERO: u32 = 0x00;
const AGC_R_DRAW_INDEX_AUTO: u32 = 0x04;
const AGC_R_DRAW_RESET: u32 = 0x05;
const AGC_R_ACB_RESET: u32 = 0x09;
const AGC_R_WAIT_MEM_32: u32 = 0x0a;
const AGC_R_SH_REGS_INDIRECT: u32 = 0x11;
const AGC_R_CX_REGS_INDIRECT: u32 = 0x12;
const AGC_R_UC_REGS_INDIRECT: u32 = 0x13;
const AGC_R_ACQUIRE_MEM: u32 = 0x14;
const AGC_R_WAIT_MEM_64: u32 = 0x16;
const AGC_R_FLIP: u32 = 0x17;
const AGC_R_RELEASE_MEM: u32 = 0x18;
const AGC_R_DMA_DATA: u32 = 0x19;
const AGC_CB_SET_SH_REGISTER_RANGE_MARKER: u32 = 0x6875_000d;

const AGC_SPI_PS_INPUT_CNTL_0: u32 = 0x191;
const AGC_VGT_PRIMITIVE_TYPE: u32 = 0x242;
const AGC_VGT_INDEX_TYPE: u32 = 0x243;
const AGC_SPI_SHADER_PGM_LO_PS: u32 = 0x008;
const AGC_SPI_SHADER_PGM_HI_PS: u32 = 0x009;
const AGC_SPI_SHADER_PGM_LO_ES: u32 = 0x0c8;
const AGC_SPI_SHADER_PGM_HI_ES: u32 = 0x0c9;
const AGC_CB_COLOR0_BASE: u32 = 0x318;
const AGC_CB_COLOR0_BASE_EXT: u32 = 0x390;
const AGC_CB_TARGET_MASK: u32 = 0x08e;
const AGC_CB_SHADER_MASK: u32 = 0x08f;
const AGC_CB_COLOR_REGISTER_STRIDE: u32 = 15;
const AGC_CB_COLOR0_VIEW: u32 = 0x31b;
const AGC_CB_COLOR0_INFO: u32 = 0x31c;
const AGC_CB_COLOR0_ATTRIB: u32 = 0x31d;
const AGC_CB_COLOR0_ATTRIB2: u32 = 0x3b0;
const AGC_CB_COLOR0_ATTRIB3: u32 = 0x3b8;
const AGC_COLOR_TARGET_COUNT: u32 = 8;
const AGC_DB_RENDER_CONTROL: u32 = 0x000;
const AGC_DB_DEPTH_CONTROL: u32 = 0x200;
const AGC_COMPUTE_USER_DATA_0: u32 = 0x240;

const AGC_SHADER_SPECIAL_GE_CNTL_OFFSET: u64 = 0x00;
const AGC_SHADER_SPECIAL_VGT_SHADER_STAGES_EN_OFFSET: u64 = 0x08;
const AGC_SHADER_SPECIAL_VGT_GS_OUT_PRIM_TYPE_OFFSET: u64 = 0x20;
const AGC_SHADER_SPECIAL_GE_USER_VGPR_EN_OFFSET: u64 = 0x28;

fn agc_create_shader(arguments: [u64; 6], state: &mut ProbeDispatchState) -> u64 {
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
    state.agc_shader_registry.headers.insert(code, header);
    state.agc_shader_registry.snapshots.remove(&code);
    state.agc_shader_registry.decode_failures.remove(&code);
    if destination != 0 {
        write_guest_u64(destination, header);
    }
    0
}

fn agc_pm4(length_dwords: u32, opcode: u32, register: u32) -> u32 {
    0xc000_0000
        | ((length_dwords.wrapping_sub(2) & 0x3fff) << 16)
        | ((opcode & 0xff) << 8)
        | ((register & 0x3f) << 2)
}

fn agc_allocate_command_dwords(command_buffer: u64, size_dwords: u32) -> Option<u64> {
    if command_buffer == 0 || size_dwords == 0 {
        return None;
    }
    let cursor_up = read_guest_u64(command_buffer + AGC_COMMAND_BUFFER_CURSOR_UP_OFFSET);
    let cursor_down = read_guest_u64(command_buffer + AGC_COMMAND_BUFFER_CURSOR_DOWN_OFFSET);
    let reserved_dwords =
        read_guest_u32(command_buffer + AGC_COMMAND_BUFFER_RESERVED_DWORDS_OFFSET);
    let available_dwords = cursor_down
        .checked_sub(cursor_up)
        .map_or(0, |bytes| bytes / 4);
    let usable_dwords = available_dwords.saturating_sub(u64::from(reserved_dwords));
    if u64::from(size_dwords) > usable_dwords {
        eprintln!(
            "[REMu][AGC] command buffer full at {command_buffer:#x}: need={size_dwords} available={usable_dwords}"
        );
        return None;
    }
    let size_bytes = u64::from(size_dwords).checked_mul(4)?;
    let next_cursor = cursor_up.checked_add(size_bytes)?;
    write_guest_u64(
        command_buffer + AGC_COMMAND_BUFFER_CURSOR_UP_OFFSET,
        next_cursor,
    );
    Some(cursor_up)
}

fn agc_dcb_reset_queue(arguments: [u64; 6]) -> u64 {
    let command_buffer = arguments[0];
    if low_u32(arguments[1]) != 0x3ff || low_u32(arguments[2]) != 0 {
        return 0;
    }
    let Some(command) = agc_allocate_command_dwords(command_buffer, 2) else {
        return 0;
    };
    write_guest_u32(command, agc_pm4(2, AGC_IT_NOP, AGC_R_DRAW_RESET));
    write_guest_u32(command + 4, 0);
    command
}

fn agc_dcb_set_registers_indirect(arguments: [u64; 6], packet_register: u32) -> u64 {
    let command_buffer = arguments[0];
    let registers = arguments[1];
    let register_count = low_u32(arguments[2]);
    let Some(command) = agc_allocate_command_dwords(command_buffer, 4) else {
        return 0;
    };
    write_guest_u32(command, agc_pm4(4, AGC_IT_NOP, packet_register));
    write_guest_u32(command + 4, register_count);
    write_guest_u32(command + 8, low_u32(registers));
    write_guest_u32(command + 12, (registers >> 32) as u32);
    command
}

fn agc_set_indirect_patch_address(arguments: [u64; 6]) -> u64 {
    let command = arguments[0];
    let registers = arguments[1];
    if command == 0 || registers == 0 {
        return u64::MAX;
    }
    write_guest_u32(command + 8, low_u32(registers));
    write_guest_u32(command + 12, (registers >> 32) as u32);
    0
}

fn agc_add_indirect_patch_registers(arguments: [u64; 6]) -> u64 {
    let command = arguments[0];
    if command == 0 {
        return u64::MAX;
    }
    let register_count = low_u32(arguments[1]);
    let current_count = read_guest_u32(command + 4);
    write_guest_u32(command + 4, current_count.wrapping_add(register_count));
    0
}

fn agc_copy_shader_register(source: u64, destination: u64) {
    write_guest_u32(destination, read_guest_u32(source));
    write_guest_u32(destination + 4, read_guest_u32(source + 4));
}

fn agc_create_primitive_state(arguments: [u64; 6]) -> u64 {
    let cx_registers = arguments[0];
    let uc_registers = arguments[1];
    let hull_shader = arguments[2];
    let geometry_shader = arguments[3];
    let primitive_type = low_u32(arguments[4]);
    if cx_registers == 0 || uc_registers == 0 || hull_shader != 0 || geometry_shader == 0 {
        return u64::MAX;
    }
    let shader_type = read_guest_u8(geometry_shader + AGC_SHADER_TYPE_OFFSET);
    if !matches!(shader_type, 2 | 6) {
        return u64::MAX;
    }
    let specials = read_guest_u64(geometry_shader + AGC_SHADER_SPECIALS_OFFSET);
    if specials == 0 {
        return u64::MAX;
    }
    agc_copy_shader_register(
        specials + AGC_SHADER_SPECIAL_VGT_SHADER_STAGES_EN_OFFSET,
        cx_registers,
    );
    agc_copy_shader_register(
        specials + AGC_SHADER_SPECIAL_VGT_GS_OUT_PRIM_TYPE_OFFSET,
        cx_registers + 8,
    );
    agc_copy_shader_register(specials + AGC_SHADER_SPECIAL_GE_CNTL_OFFSET, uc_registers);
    agc_copy_shader_register(
        specials + AGC_SHADER_SPECIAL_GE_USER_VGPR_EN_OFFSET,
        uc_registers + 8,
    );
    write_guest_u32(uc_registers + 16, AGC_VGT_PRIMITIVE_TYPE);
    write_guest_u32(uc_registers + 20, primitive_type);
    0
}

fn agc_create_interpolant_mapping(arguments: [u64; 6]) -> u64 {
    let registers = arguments[0];
    let geometry_shader = arguments[1];
    let pixel_shader = arguments[2];
    if registers == 0 || geometry_shader == 0 {
        return u64::MAX;
    }
    let output_semantics = read_guest_u64(geometry_shader + AGC_SHADER_OUTPUT_SEMANTICS_OFFSET);
    let output_count = read_guest_u32(geometry_shader + AGC_SHADER_OUTPUT_SEMANTICS_COUNT_OFFSET);
    let input_semantics = if pixel_shader == 0 {
        0
    } else {
        // Reading the count matches the API's descriptor validation even though only the pointer
        // is needed to construct the fixed 32-register mapping.
        let _ = read_guest_u32(pixel_shader + AGC_SHADER_INPUT_SEMANTICS_COUNT_OFFSET);
        read_guest_u64(pixel_shader + AGC_SHADER_INPUT_SEMANTICS_OFFSET)
    };
    for index in 0_u32..32 {
        let mut value = 0;
        if index < output_count && output_semantics != 0 {
            let flat = input_semantics != 0
                && (read_guest_u32(input_semantics + u64::from(index) * 4) >> 22) & 1 != 0;
            value = index | if flat { 0x400 } else { 0 };
        }
        let destination = registers + u64::from(index) * 8;
        write_guest_u32(destination, AGC_SPI_PS_INPUT_CNTL_0 + index);
        write_guest_u32(destination + 4, value);
    }
    0
}

fn agc_cb_set_sh_register_range_direct(arguments: [u64; 6]) -> u64 {
    let command_buffer = arguments[0];
    let offset = low_u32(arguments[1]);
    let values = arguments[2];
    let value_count = low_u32(arguments[3]);
    if command_buffer == 0 || offset == 0 || offset > 0x3ff || value_count == 0 {
        return 0;
    }
    let Some(marker) = agc_allocate_command_dwords(command_buffer, 2) else {
        return 0;
    };
    write_guest_u32(marker, agc_pm4(2, AGC_IT_NOP, AGC_R_ZERO));
    write_guest_u32(marker + 4, AGC_CB_SET_SH_REGISTER_RANGE_MARKER);
    let Some(command) = agc_allocate_command_dwords(command_buffer, value_count.saturating_add(2))
    else {
        return 0;
    };
    write_guest_u32(
        command,
        agc_pm4(value_count.saturating_add(2), AGC_IT_SET_SH_REG, 0),
    );
    write_guest_u32(command + 4, offset);
    for index in 0..value_count {
        let value = if values == 0 {
            0
        } else {
            read_guest_u32(values + u64::from(index) * 4)
        };
        write_guest_u32(command + 8 + u64::from(index) * 4, value);
    }
    command
}

fn agc_get_data_packet_payload_address(arguments: [u64; 6]) -> u64 {
    let output = arguments[0];
    let command = arguments[1];
    if output == 0 || command == 0 {
        return u64::MAX;
    }
    let payload = if low_i32(arguments[2]) == 0 {
        let header = read_guest_u32(command);
        if header & 0x3fff_0000 == 0x3fff_0000 {
            0
        } else {
            command + 4
        }
    } else {
        command + 8
    };
    write_guest_u64(output, payload);
    0
}

fn agc_dcb_event_write(arguments: [u64; 6]) -> u64 {
    let event_type = low_u32(arguments[1]) & 0xff;
    if event_type > 0x3f || arguments[2] != 0 {
        return 0;
    }
    let Some(command) = agc_allocate_command_dwords(arguments[0], 2) else {
        return 0;
    };
    write_guest_u32(command, agc_pm4(2, AGC_IT_EVENT_WRITE, 0));
    write_guest_u32(command + 4, event_type);
    command
}

fn agc_cb_nop(arguments: [u64; 6]) -> u64 {
    let dword_count = low_u32(arguments[1]);
    if !(2..=0x4001).contains(&dword_count) {
        return 0;
    }
    let Some(command) = agc_allocate_command_dwords(arguments[0], dword_count) else {
        return 0;
    };
    write_guest_u32(command, agc_pm4(dword_count, AGC_IT_NOP, AGC_R_ZERO));
    for index in 1..dword_count {
        write_guest_u32(command + u64::from(index) * 4, 0);
    }
    command
}

fn agc_dcb_dma_data(arguments: [u64; 6], stack_arguments: [u64; 6]) -> u64 {
    let byte_count = low_u32(stack_arguments[2]);
    if arguments[0] == 0 || byte_count == 0 || byte_count & 3 != 0 {
        return 0;
    }
    let Some(command) = agc_allocate_command_dwords(arguments[0], 8) else {
        return 0;
    };
    let destination = low_u32(arguments[1]) & 0xff;
    let destination_cache_policy = low_u32(arguments[2]) & 0xff;
    let source = low_u32(arguments[3]) & 0xff;
    let source_cache_policy = low_u32(arguments[5]) & 0xff;
    let control4 = low_u32(stack_arguments[0]) & 0xff;
    let source_address = stack_arguments[1];
    let control7 = low_u32(stack_arguments[3]) & 0xff;
    let control8 = low_u32(stack_arguments[4]) & 0xff;
    let control9 = low_u32(stack_arguments[5]) & 0xff;
    write_guest_u32(command, agc_pm4(8, AGC_IT_NOP, AGC_R_DMA_DATA));
    write_guest_u32(
        command + 4,
        destination
            | (destination_cache_policy << 8)
            | (source << 16)
            | (source_cache_policy << 24),
    );
    write_guest_u32(
        command + 8,
        control4 | (control7 << 8) | (control8 << 16) | (control9 << 24),
    );
    write_guest_u32(command + 12, byte_count);
    write_guest_u64(command + 16, arguments[4]);
    write_guest_u64(command + 24, source_address);
    command
}

fn agc_cb_release_mem(arguments: [u64; 6], stack_arguments: [u64; 6]) -> u64 {
    let destination = low_u32(arguments[3]) & 0xff;
    let data_selection = low_u32(stack_arguments[0]) & 0xff;
    let gds_offset = low_u32(stack_arguments[2]) & 0xffff;
    let gds_size = low_u32(stack_arguments[3]) & 0xffff;
    let interrupt = low_u32(stack_arguments[4]) & 0xff;
    if arguments[0] == 0
        || destination > 1
        || data_selection > 3
        || gds_offset != 0
        || gds_size > 2
        || interrupt > 3
    {
        return 0;
    }
    let Some(command) = agc_allocate_command_dwords(arguments[0], 8) else {
        return 0;
    };
    let action = low_u32(arguments[1]) & 0xff;
    let gcr_control = low_u32(arguments[2]) & 0xffff;
    let cache_policy = low_u32(arguments[4]) & 0xff;
    let destination_address = arguments[5];
    let data = stack_arguments[1];
    write_guest_u32(command, agc_pm4(8, AGC_IT_NOP, AGC_R_RELEASE_MEM));
    write_guest_u32(command + 4, action | (cache_policy << 8));
    write_guest_u32(
        command + 8,
        gcr_control | (data_selection << 16) | (interrupt << 24),
    );
    write_guest_u32(command + 12, low_u32(destination_address));
    write_guest_u32(command + 16, (destination_address >> 32) as u32);
    write_guest_u32(command + 20, low_u32(data));
    write_guest_u32(command + 24, (data >> 32) as u32);
    write_guest_u32(command + 28, low_u32(stack_arguments[5]));
    command
}

fn agc_dcb_wait_reg_mem(arguments: [u64; 6], stack_arguments: [u64; 6]) -> u64 {
    let size = low_u32(arguments[1]) & 0xff;
    let compare_function = low_u32(arguments[2]) & 0xff;
    let operation = low_u32(arguments[3]) & 0xff;
    let cache_policy = low_u32(arguments[4]) & 0xff;
    if arguments[0] == 0 || size > 1 || compare_function > 7 || operation > 4 || cache_policy > 3 {
        return 0;
    }
    let standard_wait = matches!(operation, 2 | 3);
    let packet_dwords = if standard_wait {
        7
    } else if size == 0 {
        6
    } else {
        9
    };
    let Some(command) = agc_allocate_command_dwords(arguments[0], packet_dwords) else {
        return 0;
    };
    let address = arguments[5];
    let reference = stack_arguments[0];
    let mask = stack_arguments[1];
    let poll_cycles = low_u32(stack_arguments[2]);
    if standard_wait {
        write_guest_u32(command, agc_pm4(packet_dwords, AGC_IT_WAIT_REG_MEM, 0));
        write_guest_u32(command + 4, compare_function | ((operation & 1) << 8));
        write_guest_u32(command + 8, low_u32(address));
        write_guest_u32(command + 12, (address >> 32) as u32);
        write_guest_u32(command + 16, low_u32(reference));
        write_guest_u32(command + 20, low_u32(mask));
        write_guest_u32(command + 24, poll_cycles / 40);
    } else {
        let packet_register = if size == 0 {
            AGC_R_WAIT_MEM_32
        } else {
            AGC_R_WAIT_MEM_64
        };
        write_guest_u32(command, agc_pm4(packet_dwords, AGC_IT_NOP, packet_register));
        write_guest_u32(command + 4, low_u32(address));
        write_guest_u32(command + 8, (address >> 32) as u32);
        write_guest_u32(command + 12, low_u32(mask));
        if size == 0 {
            write_guest_u32(command + 16, compare_function | (operation << 8));
            write_guest_u32(command + 20, low_u32(reference));
        } else {
            write_guest_u32(command + 16, (mask >> 32) as u32);
            write_guest_u32(command + 20, low_u32(reference));
            write_guest_u32(command + 24, (reference >> 32) as u32);
            write_guest_u32(command + 28, compare_function | (operation << 8));
            write_guest_u32(command + 32, poll_cycles / 40);
        }
    }
    command
}

fn agc_packet_identity(command: u64) -> Option<(u32, u32, u32)> {
    if command == 0 {
        return None;
    }
    let header = read_guest_u32(command);
    if header >> 30 != 3 {
        return None;
    }
    Some((
        ((header >> 8) & 0xff),
        (header >> 2) & 0x3f,
        ((header >> 16) & 0x3fff) + 2,
    ))
}

fn agc_dma_data_patch_destination(arguments: [u64; 6]) -> u64 {
    let command = arguments[0];
    let destination = arguments[1];
    let Some((opcode, register, length)) = agc_packet_identity(command) else {
        return u64::MAX;
    };
    if opcode != AGC_IT_NOP || register != AGC_R_DMA_DATA {
        return u64::MAX;
    }
    let destination_offset = if length == 7 { 4 } else { 16 };
    write_guest_u64(command + destination_offset, destination);
    0
}

fn agc_wait_reg_mem_patch_address(arguments: [u64; 6]) -> u64 {
    let command = arguments[0];
    let address = arguments[1];
    let Some((opcode, register, _)) = agc_packet_identity(command) else {
        return u64::MAX;
    };
    let offset = if opcode == AGC_IT_WAIT_REG_MEM {
        8
    } else if opcode == AGC_IT_NOP && matches!(register, AGC_R_WAIT_MEM_32 | AGC_R_WAIT_MEM_64) {
        4
    } else {
        return u64::MAX;
    };
    write_guest_u64(command + offset, address);
    0
}

fn agc_release_mem_patch_address(arguments: [u64; 6]) -> u64 {
    let command = arguments[0];
    let Some((opcode, register, _)) = agc_packet_identity(command) else {
        return u64::MAX;
    };
    if opcode != AGC_IT_NOP || register != AGC_R_RELEASE_MEM {
        return u64::MAX;
    }
    write_guest_u64(command + 12, arguments[1]);
    0
}

fn agc_dcb_acquire_mem(arguments: [u64; 6], stack_arguments: [u64; 6]) -> u64 {
    let command_buffer = arguments[0];
    let engine = low_u32(arguments[1]) & 0xff;
    let cb_db_operation = low_u32(arguments[2]);
    let gcr_control = low_u32(arguments[3]);
    let base_address = arguments[4];
    let size_bytes = arguments[5];
    let no_size = size_bytes == u64::MAX;
    if command_buffer == 0
        || engine > 1
        || (!no_size && (size_bytes & 0xff != 0 || size_bytes >> 40 != 0))
        || base_address & 0xff != 0
        || base_address >> 40 != 0
    {
        return 0;
    }
    let Some(command) = agc_allocate_command_dwords(command_buffer, 8) else {
        return 0;
    };
    write_guest_u32(command, agc_pm4(8, AGC_IT_NOP, AGC_R_ACQUIRE_MEM));
    write_guest_u32(command + 4, (engine << 31) | cb_db_operation);
    write_guest_u32(
        command + 8,
        if no_size { 0 } else { low_u32(size_bytes >> 8) },
    );
    write_guest_u32(command + 12, 0);
    write_guest_u32(command + 16, low_u32(base_address >> 8));
    write_guest_u32(command + 20, 0);
    write_guest_u32(command + 24, low_u32(stack_arguments[0]) / 40);
    write_guest_u32(command + 28, gcr_control);
    command
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
        return sce_kernel_error(VIDEO_OUT_ERROR_INVALID_ADDRESS);
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

const VIDEO_OUT_ERROR_INVALID_VALUE: u32 = 0x8029_0001;
const VIDEO_OUT_ERROR_INVALID_ADDRESS: u32 = 0x8029_0002;
const VIDEO_OUT_ERROR_RESOURCE_BUSY: u32 = 0x8029_0009;
const VIDEO_OUT_ERROR_INVALID_INDEX: u32 = 0x8029_000a;
const VIDEO_OUT_ERROR_INVALID_HANDLE: u32 = 0x8029_000b;

fn video_out_open(arguments: [u64; 6], state: &mut ProbeDispatchState) -> u64 {
    let user_id = low_i32(arguments[0]);
    let bus_type = low_i32(arguments[1]);
    let index = low_i32(arguments[2]);
    if bus_type != 0 || index != 0 {
        return sce_kernel_error(VIDEO_OUT_ERROR_INVALID_VALUE);
    }
    if state.video_out.is_open {
        return sce_kernel_error(VIDEO_OUT_ERROR_RESOURCE_BUSY);
    }
    state.video_out.is_open = true;
    state.video_out.user_id = user_id;
    state.video_out.bus_type = bus_type;
    state.video_out.index = index;
    1
}

fn video_out_register_buffers2(arguments: [u64; 6], state: &mut ProbeDispatchState) -> u64 {
    let set_index = low_u32(arguments[1]);
    let buffer_index_start = usize::try_from(low_u32(arguments[2])).unwrap_or(usize::MAX);
    let entries = arguments[3];
    let buffer_count = usize::try_from(low_u32(arguments[4])).unwrap_or(usize::MAX);
    let attribute = arguments[5];
    if low_i32(arguments[0]) != 1 || !state.video_out.is_open {
        return sce_kernel_error(VIDEO_OUT_ERROR_INVALID_HANDLE);
    }
    if entries == 0 || attribute == 0 {
        return sce_kernel_error(VIDEO_OUT_ERROR_INVALID_ADDRESS);
    }
    if buffer_index_start > 15
        || buffer_count == 0
        || buffer_count > 16
        || buffer_index_start.saturating_add(buffer_count) > 16
    {
        return sce_kernel_error(VIDEO_OUT_ERROR_INVALID_INDEX);
    }

    state.video_out.pixel_format = read_guest_u64(attribute + 0x20);
    state.video_out.tiling_mode = read_guest_u32(attribute + 0x04);
    state.video_out.width = read_guest_u32(attribute + 0x0c);
    state.video_out.height = read_guest_u32(attribute + 0x10);
    state
        .video_out
        .display_buffers
        .resize(buffer_index_start + buffer_count, 0);
    state
        .video_out
        .display_buffer_attributes
        .resize(buffer_index_start + buffer_count, None);
    let attributes = VideoOutBufferAttributes {
        width: state.video_out.width,
        height: state.video_out.height,
        pitch: state.video_out.width,
        pixel_format: state.video_out.pixel_format,
        tiling_mode: state.video_out.tiling_mode,
    };
    for index in 0..buffer_count {
        let entry = entries + u64::try_from(index).unwrap_or(u64::MAX) * 0x20;
        state.video_out.display_buffers[buffer_index_start + index] = read_guest_u64(entry);
        state.video_out.display_buffer_attributes[buffer_index_start + index] = Some(attributes);
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
    remu_gpu::emit_event(remu_gpu::GpuEvent::DisplayRegistered {
        width: state.video_out.width,
        height: state.video_out.height,
        pitch: state.video_out.width,
        pixel_format: state.video_out.pixel_format,
        tiling_mode: state.video_out.tiling_mode,
        buffers: state.video_out.display_buffers.clone(),
    });
    u64::from(set_index)
}

fn video_out_add_flip_event(arguments: [u64; 6], state: &mut ProbeDispatchState) -> u64 {
    let equeue = arguments[0];
    let handle = low_i32(arguments[1]);
    if handle != 1 || !state.video_out.is_open {
        return sce_kernel_error(VIDEO_OUT_ERROR_INVALID_HANDLE);
    }
    if equeue == 0 {
        return sce_kernel_error(VIDEO_OUT_ERROR_INVALID_ADDRESS);
    }
    state.video_out.flip_events.insert(equeue, arguments[2]);
    0
}

fn agc_dcb_set_flip(arguments: [u64; 6]) -> u64 {
    let command_buffer = arguments[0];
    if command_buffer == 0 {
        return 0;
    }
    let Some(command) = agc_allocate_command_dwords(command_buffer, 6) else {
        return 0;
    };
    write_guest_u32(command, agc_pm4(6, AGC_IT_NOP, AGC_R_FLIP));
    write_guest_u32(command + 4, low_u32(arguments[1]));
    write_guest_u32(command + 8, low_u32(arguments[2]));
    write_guest_u32(command + 12, low_u32(arguments[3]));
    write_guest_u32(command + 16, low_u32(arguments[4]));
    write_guest_u32(command + 20, low_u32(arguments[4] >> 32));
    command
}

fn agc_dcb_set_index_buffer(arguments: [u64; 6], state: &mut ProbeDispatchState) -> u64 {
    let Some(command) = agc_allocate_command_dwords(arguments[0], 5) else {
        return 0;
    };
    // Gen5's four-argument form carries a CPU-readable index pointer, its GPU resource address,
    // and the element capacity. Keep the association so the host renderer can read the indices
    // while preserving the resource address needed for later descriptor translation.
    let index_buffer_address = arguments[1];
    let index_resource_address = arguments[2];
    let index_count = low_u32(arguments[3]);
    state
        .agc_index_resources
        .insert(index_buffer_address, index_resource_address);
    write_guest_u32(command, agc_pm4(3, AGC_IT_INDEX_BASE, 0));
    write_guest_u32(command + 4, low_u32(index_buffer_address));
    write_guest_u32(command + 8, low_u32(index_buffer_address >> 32));
    write_guest_u32(command + 12, agc_pm4(2, AGC_IT_INDEX_BUFFER_SIZE, 0));
    write_guest_u32(command + 16, index_count);
    command
}

fn agc_dcb_set_index_size(arguments: [u64; 6]) -> u64 {
    if low_u32(arguments[2]) != 0 {
        return 0;
    }
    let Some(command) = agc_allocate_command_dwords(arguments[0], 2) else {
        return 0;
    };
    write_guest_u32(command, agc_pm4(2, AGC_IT_INDEX_TYPE, 0));
    write_guest_u32(command + 4, low_u32(arguments[1]) & 0xff);
    command
}

fn agc_dcb_set_num_instances(arguments: [u64; 6]) -> u64 {
    let Some(command) = agc_allocate_command_dwords(arguments[0], 2) else {
        return 0;
    };
    write_guest_u32(command, agc_pm4(2, AGC_IT_NUM_INSTANCES, 0));
    write_guest_u32(command + 4, low_u32(arguments[1]));
    command
}

fn agc_dcb_draw_index_auto(arguments: [u64; 6]) -> u64 {
    if low_u32(arguments[2]) != 0x4000_0000 {
        return 0;
    }
    let Some(command) = agc_allocate_command_dwords(arguments[0], 7) else {
        return 0;
    };
    write_guest_u32(command, agc_pm4(7, AGC_IT_NOP, AGC_R_DRAW_INDEX_AUTO));
    write_guest_u32(command + 4, low_u32(arguments[1]));
    for index in 2..7_u64 {
        write_guest_u32(command + index * 4, 0);
    }
    command
}

fn agc_dcb_draw_index_offset(arguments: [u64; 6]) -> u64 {
    let Some(command) = agc_allocate_command_dwords(arguments[0], 5) else {
        return 0;
    };
    write_guest_u32(command, agc_pm4(5, AGC_IT_DRAW_INDEX_OFFSET, 0));
    write_guest_u32(command + 4, low_u32(arguments[2]));
    write_guest_u32(command + 8, low_u32(arguments[1]));
    write_guest_u32(command + 12, low_u32(arguments[2]));
    write_guest_u32(command + 16, low_u32(arguments[3]) & 0xe000_0001);
    command
}

fn queue_video_out_flip(
    state: &mut ProbeDispatchState,
    handle: u32,
    buffer_index: u32,
    flip_argument: u64,
) {
    let Some(&display_address) = state.video_out.display_buffers.get(buffer_index as usize) else {
        eprintln!("[REMu][VIDEO] ignored flip for unregistered buffer index {buffer_index}");
        return;
    };
    let event_data = 6 | ((flip_argument & 0x0000_ffff_ffff_ffff) << 16);
    for (&equeue, &user_data) in &state.video_out.flip_events {
        state.pending_events.push_back(KernelQueuedEvent {
            equeue,
            ident: 6,
            filter: -13,
            flags: 0,
            fflags: 0,
            data: event_data,
            user_data,
        });
        if state.agc_submission_count <= 8 {
            eprintln!(
                "[REMu][VIDEO] queued flip handle={handle} index={buffer_index} address={display_address:#x} equeue={equeue:#x}"
            );
        }
    }
}

fn agc_submit_ordered_video_out_flip(
    graphics: &mut AgcGraphicsState,
    video_out: &VideoOutState,
    handle: u32,
    buffer_index: u32,
    flip_argument: u64,
) -> bool {
    if handle != 1 || !video_out.is_open {
        eprintln!("[REMu][VIDEO] ignored flip for invalid handle {handle}");
        graphics.pending_targetless_draw = None;
        return false;
    }
    let Some(&display_address) = video_out.display_buffers.get(buffer_index as usize) else {
        eprintln!("[REMu][VIDEO] ignored flip for unregistered buffer index {buffer_index}");
        return false;
    };
    let Some(attributes) = video_out
        .display_buffer_attributes
        .get(buffer_index as usize)
        .copied()
        .flatten()
    else {
        eprintln!("[REMu][VIDEO] ignored flip without attributes for buffer index {buffer_index}");
        graphics.pending_targetless_draw = None;
        return false;
    };
    if let Some(mut draw) = graphics.pending_targetless_draw.take() {
        let target = graphics
            .known_color_targets
            .get(&display_address)
            .copied()
            .filter(|target| {
                ((target.attrib2 >> 14) & 0x3fff) + 1 == attributes.width
                    && (target.attrib2 & 0x3fff) + 1 == attributes.height
            });
        let export_masks = draw
            .pixel_shader
            .as_ref()
            .map_or(0, |shader| shader.program.pixel_color_export_masks())
            & draw
                .context_registers
                .iter()
                .find(|register| register.offset == AGC_CB_SHADER_MASK)
                .map_or(u32::MAX, |register| register.value);
        let export_slots = (0..AGC_COLOR_TARGET_COUNT)
            .filter(|slot| (export_masks >> (slot * 4)) & 0xf != 0)
            .collect::<Vec<_>>();
        if let [export_slot] = export_slots.as_slice() {
            draw.scanout_target = Some(remu_gpu::GpuScanoutTarget {
                address: display_address,
                width: attributes.width,
                height: attributes.height,
                pitch: attributes.pitch,
                pixel_format: attributes.pixel_format,
                tiling_mode: attributes.tiling_mode,
                export_slot: *export_slot,
                write_mask: (export_masks >> (*export_slot * 4)) & 0xf,
                color_surface: target,
            });
            if draw.sequence <= 8 {
                eprintln!(
                    "[REMu][AGC] resolved targetless draw={} against scanout target={display_address:#x} programmed_surface={}",
                    draw.sequence,
                    target.is_some(),
                );
            }
            remu_gpu::emit_event(remu_gpu::GpuEvent::IndexedDraw(draw));
        } else if draw.sequence <= 8 {
            eprintln!(
                "[REMu][AGC] discarded targetless draw={}: flip target {display_address:#x} has ambiguous pixel exports {export_slots:?}",
                draw.sequence,
            );
        }
    }
    remu_gpu::emit_event(remu_gpu::GpuEvent::Flip {
        address: display_address,
        buffer_index,
        flip_argument,
        width: attributes.width,
        height: attributes.height,
        pitch: attributes.pitch,
        pixel_format: attributes.pixel_format,
        tiling_mode: attributes.tiling_mode,
    });
    true
}

fn agc_apply_submission_registers(
    packet_address: u64,
    length: u32,
    opcode: u32,
    register: u32,
    graphics: &mut AgcGraphicsState,
) -> Result<(), AgcSubmissionError> {
    if opcode == AGC_IT_SET_UCONFIG_REG_INDEX {
        if length != 3 {
            return Err(AgcSubmissionError::IndexedUconfigRegisterPacketLength { actual: length });
        }
        let register = read_guest_u32(packet_address + 4) & !0x7000_0000;
        let value = read_guest_u32(packet_address + 8);
        graphics.user_config_registers.insert(register, value);
        if register == AGC_VGT_INDEX_TYPE {
            graphics.index_size = value & 0x3;
        }
        return Ok(());
    }

    if matches!(
        opcode,
        AGC_IT_SET_CONTEXT_REG | AGC_IT_SET_SH_REG | AGC_IT_SET_UCONFIG_REG
    ) {
        if length < 3 {
            return Ok(());
        }
        let first_register = read_guest_u32(packet_address + 4);
        let destination = match opcode {
            AGC_IT_SET_CONTEXT_REG => &mut graphics.context_registers,
            AGC_IT_SET_SH_REG => &mut graphics.shader_registers,
            _ => &mut graphics.user_config_registers,
        };
        for index in 0..length - 2 {
            destination.insert(
                first_register + index,
                read_guest_u32(packet_address + u64::from(index + 2) * 4),
            );
        }
        return Ok(());
    }

    if matches!(
        opcode,
        AGC_IT_SET_CONTEXT_REG_INDIRECT
            | AGC_IT_SET_SH_REG_INDIRECT
            | AGC_IT_SET_UCONFIG_REG_INDIRECT
    ) {
        if length != 5 {
            return Err(AgcSubmissionError::NativeIndirectRegisterPacketLength {
                opcode,
                actual: length,
            });
        }
        let registers_address = u64::from(read_guest_u32(packet_address + 4) & !0x3)
            | (u64::from(read_guest_u32(packet_address + 8)) << 32);
        let register_count = read_guest_u32(packet_address + 16) & 0x3fff;
        let (destination, normalize_selector) = match opcode {
            AGC_IT_SET_CONTEXT_REG_INDIRECT => (&mut graphics.context_registers, false),
            AGC_IT_SET_SH_REG_INDIRECT => (&mut graphics.shader_registers, true),
            _ => (&mut graphics.user_config_registers, true),
        };
        agc_apply_indirect_register_table(
            opcode,
            registers_address,
            register_count,
            normalize_selector,
            destination,
        )?;
        return Ok(());
    }

    if opcode != AGC_IT_NOP
        || !matches!(
            register,
            AGC_R_SH_REGS_INDIRECT | AGC_R_CX_REGS_INDIRECT | AGC_R_UC_REGS_INDIRECT
        )
        || length < 4
    {
        return Ok(());
    }
    let register_count = read_guest_u32(packet_address + 4).min(4_096);
    let registers_address = read_guest_u64(packet_address + 8);
    let destination = match register {
        AGC_R_CX_REGS_INDIRECT => &mut graphics.context_registers,
        AGC_R_SH_REGS_INDIRECT => &mut graphics.shader_registers,
        _ => &mut graphics.user_config_registers,
    };
    agc_apply_indirect_register_table(
        opcode,
        registers_address,
        register_count,
        false,
        destination,
    )
}

fn agc_apply_indirect_register_table(
    opcode: u32,
    registers_address: u64,
    register_count: u32,
    normalize_selector: bool,
    destination: &mut BTreeMap<u32, u32>,
) -> Result<(), AgcSubmissionError> {
    if register_count == 0 {
        return Ok(());
    }
    if registers_address == 0 {
        return Err(AgcSubmissionError::NullIndirectRegisterTable {
            opcode,
            register_count,
        });
    }
    let byte_count =
        usize::try_from(register_count).expect("14-bit indirect register count fits usize") * 8;
    let mut bytes = vec![0_u8; byte_count];
    if registers_address
        .checked_add(u64::try_from(byte_count).expect("register table length fits u64"))
        .is_none()
        || !ProcessGuestMemory.read(registers_address, &mut bytes)
    {
        return Err(AgcSubmissionError::UnreadableIndirectRegister(
            registers_address,
        ));
    }
    for pair in bytes.chunks_exact(8) {
        let raw_offset = u32::from_le_bytes(pair[..4].try_into().expect("register offset dword"));
        if raw_offset == u32::MAX {
            continue;
        }
        let offset = if normalize_selector {
            raw_offset & !0x7000_0000
        } else {
            raw_offset
        };
        let value = u32::from_le_bytes(pair[4..].try_into().expect("register value dword"));
        destination.insert(offset, value);
    }
    Ok(())
}

fn agc_program_address(registers: &BTreeMap<u32, u32>, low: u32, high: u32) -> u64 {
    let Some((&low, &high)) = registers.get(&low).zip(registers.get(&high)) else {
        return 0;
    };
    (u64::from(high & 0xff) << 40) | (u64::from(low) << 8)
}

#[derive(Debug)]
enum AgcColorTargetState {
    Absent,
    Programmed(Vec<remu_gpu::GpuColorTarget>),
    Malformed { slot: u32 },
}

fn agc_color_target_state(registers: &BTreeMap<u32, u32>) -> AgcColorTargetState {
    let target_mask = registers.get(&AGC_CB_TARGET_MASK).copied();
    let mut targets = Vec::new();
    for slot in 0..AGC_COLOR_TARGET_COUNT {
        if target_mask.is_some_and(|mask| (mask >> (slot * 4)).trailing_zeros() >= 4) {
            continue;
        }
        let primary = slot * AGC_CB_COLOR_REGISTER_STRIDE;
        let base = registers.get(&(AGC_CB_COLOR0_BASE + primary)).copied();
        let base_ext = registers.get(&(AGC_CB_COLOR0_BASE_EXT + slot)).copied();
        let address = base
            .zip(base_ext)
            .map(|(low, high)| (u64::from(high & 0xff) << 40) | (u64::from(low) << 8));
        let explicitly_enabled =
            target_mask.is_some_and(|mask| (mask >> (slot * 4)).trailing_zeros() < 4);
        if address.is_none_or(|address| address == 0) {
            if explicitly_enabled {
                return AgcColorTargetState::Malformed { slot };
            }
            continue;
        }
        let Some(view) = registers.get(&(AGC_CB_COLOR0_VIEW + primary)).copied() else {
            return AgcColorTargetState::Malformed { slot };
        };
        let Some(info) = registers.get(&(AGC_CB_COLOR0_INFO + primary)).copied() else {
            return AgcColorTargetState::Malformed { slot };
        };
        let Some(attrib) = registers.get(&(AGC_CB_COLOR0_ATTRIB + primary)).copied() else {
            return AgcColorTargetState::Malformed { slot };
        };
        let Some(attrib2) = registers.get(&(AGC_CB_COLOR0_ATTRIB2 + slot)).copied() else {
            return AgcColorTargetState::Malformed { slot };
        };
        let Some(attrib3) = registers.get(&(AGC_CB_COLOR0_ATTRIB3 + slot)).copied() else {
            return AgcColorTargetState::Malformed { slot };
        };
        targets.push(remu_gpu::GpuColorTarget {
            slot,
            address: address.expect("nonzero color target address was checked"),
            view,
            info,
            attrib,
            attrib2,
            attrib3,
        });
    }
    if targets.is_empty() {
        AgcColorTargetState::Absent
    } else {
        AgcColorTargetState::Programmed(targets)
    }
}

fn agc_has_active_depth_state(registers: &BTreeMap<u32, u32>) -> bool {
    registers
        .get(&AGC_DB_DEPTH_CONTROL)
        .is_some_and(|control| control & 0x6 != 0)
        || registers
            .get(&AGC_DB_RENDER_CONTROL)
            .is_some_and(|control| control & 1 != 0)
}

fn agc_register_snapshot(registers: &BTreeMap<u32, u32>) -> Vec<remu_gpu::GpuRegister> {
    registers
        .iter()
        .map(|(&offset, &value)| remu_gpu::GpuRegister { offset, value })
        .collect()
}

fn agc_shader_prefix_snapshot(address: u64) -> [u32; 4] {
    if address == 0 {
        return [0; 4];
    }
    std::array::from_fn(|index| {
        read_guest_u32(address + u64::try_from(index).expect("shader prefix index fits") * 4)
    })
}

struct ProcessGuestMemory;

impl GuestMemoryReader for ProcessGuestMemory {
    fn read(&self, address: u64, output: &mut [u8]) -> bool {
        if output.is_empty() {
            return true;
        }
        #[cfg(target_os = "macos")]
        {
            let Ok(length) = u64::try_from(output.len()) else {
                return false;
            };
            let mut copied = 0_u64;
            // SAFETY: mach_vm_read_overwrite validates the source mapping in the current task and
            // writes at most `length` bytes to the live output slice.
            let result = unsafe {
                mach_vm_read_overwrite(
                    mach_task_self(),
                    address,
                    length,
                    output.as_mut_ptr() as u64,
                    &raw mut copied,
                )
            };
            result == 0 && copied == length
        }
        #[cfg(target_os = "linux")]
        {
            let local = libc::iovec {
                iov_base: output.as_mut_ptr().cast(),
                iov_len: output.len(),
            };
            let remote = libc::iovec {
                iov_base: usize::try_from(address)
                    .map_or(std::ptr::null_mut(), |value| value as *mut libc::c_void),
                iov_len: output.len(),
            };
            if remote.iov_base.is_null() {
                return false;
            }
            // SAFETY: both iovecs describe live output storage and a source in this same process;
            // process_vm_readv returns EFAULT instead of dereferencing an unreadable guest range.
            let copied =
                unsafe { libc::process_vm_readv(libc::getpid(), &local, 1, &remote, 1, 0) };
            copied == isize::try_from(output.len()).unwrap_or(-1)
        }
        #[cfg(not(any(target_os = "linux", target_os = "macos")))]
        {
            let _ = address;
            false
        }
    }
}

fn agc_shader_snapshot(
    shaders: &mut AgcShaderRegistry,
    address: u64,
) -> Option<remu_gpu::ShaderSnapshot> {
    if address == 0 || shaders.decode_failures.contains(&address) {
        return None;
    }
    if let Some(snapshot) = shaders.snapshots.get(&address) {
        return Some(snapshot.clone());
    }
    let memory = ProcessGuestMemory;
    let program = match remu_shader::read_program(&memory, address) {
        Ok(program) => Arc::new(program),
        Err(error) => {
            shaders.decode_failures.insert(address);
            eprintln!("[REMu][AGC][SHADER] decode failed at {address:#x}: {error}");
            return None;
        }
    };
    let header_address = shaders.headers.get(&address).copied().unwrap_or_default();
    let metadata = (header_address != 0)
        .then(|| remu_shader::read_shader_metadata(&memory, header_address))
        .transpose()
        .map_err(|error| {
            eprintln!("[REMu][AGC][SHADER] metadata failed at header {header_address:#x}: {error}");
        })
        .ok()
        .flatten();
    let snapshot = remu_gpu::ShaderSnapshot {
        header_address,
        program,
        metadata,
    };
    shaders.snapshots.insert(address, snapshot.clone());
    Some(snapshot)
}

#[derive(Debug, Error)]
enum AgcComputeDispatchError {
    #[error(transparent)]
    Decode(#[from] remu_gpu::ComputeDispatchDecodeError),
    #[error("base-relative DISPATCH_INDIRECT has no active compute indirect-argument base")]
    MissingIndirectBase,
    #[error("compute indirect dimensions address overflows the guest address space")]
    IndirectAddressOverflow,
    #[error("compute indirect dimensions at 0x{0:016X} are not readable")]
    UnreadableIndirectDimensions(u64),
    #[error("compute program registers COMPUTE_PGM_LO/HI are incomplete")]
    MissingProgramAddress,
    #[error("compute program address is null")]
    NullProgramAddress,
    #[error("compute shader at 0x{0:016X} could not be decoded")]
    ShaderDecode(u64),
    #[error("compute shader at 0x{address:016X} could not be evaluated: {reason}")]
    ShaderEvaluation { address: u64, reason: String },
}

#[derive(Debug, Error)]
enum AgcSubmissionError {
    #[error("unsupported PM4 packet type {packet_type} at dword {dword} (header 0x{header:08X})")]
    UnsupportedPacketType {
        dword: u32,
        packet_type: u32,
        header: u32,
    },
    #[error("PM4 packet at dword {dword} declares {length} dwords with {remaining} remaining")]
    InvalidPacketLength {
        dword: u32,
        length: u32,
        remaining: u32,
    },
    #[error("SET_BASE packet has {actual} dwords, expected 4")]
    SetBasePacketLength { actual: u32 },
    #[error("SET_BASE selects unsupported base index {0}")]
    UnsupportedBaseIndex(u32),
    #[error("SET_BASE selects unsupported shader type {0}")]
    UnsupportedSetBaseShaderType(u32),
    #[error("CLEAR_STATE packet has {actual} dwords, expected 2")]
    ClearStatePacketLength { actual: u32 },
    #[error("CLEAR_STATE control has unsupported bits 0x{0:08X}")]
    UnsupportedClearStateControl(u32),
    #[error("native indirect-register opcode 0x{opcode:02X} has {actual} dwords, expected 5")]
    NativeIndirectRegisterPacketLength { opcode: u32, actual: u32 },
    #[error("indexed SET_UCONFIG_REG packet has {actual} dwords, expected 3")]
    IndexedUconfigRegisterPacketLength { actual: u32 },
    #[error(
        "indirect-register opcode 0x{opcode:02X} has null table address with {register_count} entries"
    )]
    NullIndirectRegisterTable { opcode: u32, register_count: u32 },
    #[error("indirect register entry at 0x{0:016X} is not readable")]
    UnreadableIndirectRegister(u64),
    #[error("ACB descriptor has address 0x{address:016X} and {dword_count} dwords")]
    InvalidAcbDescriptor { address: u64, dword_count: u32 },
    #[error("compute dispatch at 0x{packet_address:016X}: {source}")]
    Compute {
        packet_address: u64,
        #[source]
        source: AgcComputeDispatchError,
    },
}

fn agc_packet_words(packet_address: u64, length: u32) -> Vec<u32> {
    (0..length)
        .map(|index| read_guest_u32(packet_address + u64::from(index) * 4))
        .collect()
}

fn agc_read_compute_dimensions(address: u64) -> Result<[u32; 3], AgcComputeDispatchError> {
    let mut bytes = [0_u8; 12];
    address
        .checked_add(u64::try_from(bytes.len()).expect("dimension byte count fits u64"))
        .ok_or(AgcComputeDispatchError::IndirectAddressOverflow)?;
    if !ProcessGuestMemory.read(address, &mut bytes) {
        return Err(AgcComputeDispatchError::UnreadableIndirectDimensions(
            address,
        ));
    }
    Ok(std::array::from_fn(|axis| {
        let start = axis * 4;
        u32::from_le_bytes(
            bytes[start..start + 4]
                .try_into()
                .expect("dimension chunk is four bytes"),
        )
    }))
}

fn agc_apply_set_base(
    packet_address: u64,
    length: u32,
    graphics: &mut AgcGraphicsState,
) -> Result<(), AgcSubmissionError> {
    if length != 4 {
        return Err(AgcSubmissionError::SetBasePacketLength { actual: length });
    }
    let base_index = read_guest_u32(packet_address + 4) & 0xf;
    if base_index != 1 {
        return Err(AgcSubmissionError::UnsupportedBaseIndex(base_index));
    }
    let header = read_guest_u32(packet_address);
    let shader_type = (header >> 1) & 0x3;
    if shader_type > 1 {
        return Err(AgcSubmissionError::UnsupportedSetBaseShaderType(
            shader_type,
        ));
    }
    if shader_type == 1 {
        let low = read_guest_u32(packet_address + 8) & !0x7;
        let high = read_guest_u32(packet_address + 12) & 0xffff;
        graphics.compute_indirect_args_address = u64::from(low) | (u64::from(high) << 32);
    }
    Ok(())
}

fn agc_decode_compute_dispatch(
    packet_address: u64,
    length: u32,
    opcode: u32,
    graphics: &mut AgcGraphicsState,
    shaders: &mut AgcShaderRegistry,
) -> Result<remu_gpu::ComputeDispatch, AgcComputeDispatchError> {
    let packet = agc_packet_words(packet_address, length);
    let shader_registers = agc_register_snapshot(&graphics.shader_registers);
    let (source, launch) = if opcode == AGC_IT_DISPATCH_DIRECT {
        (
            remu_gpu::ComputeDispatchSource::Direct,
            remu_gpu::decode_compute_dispatch_direct(&packet, &shader_registers)?,
        )
    } else {
        let indirect = remu_gpu::decode_compute_dispatch_indirect(&packet)?;
        let dimensions_address = match indirect.dimensions {
            remu_gpu::ComputeIndirectDimensions::BaseOffset(offset) => {
                if graphics.compute_indirect_args_address == 0 {
                    return Err(AgcComputeDispatchError::MissingIndirectBase);
                }
                graphics
                    .compute_indirect_args_address
                    .checked_add(u64::from(offset))
                    .ok_or(AgcComputeDispatchError::IndirectAddressOverflow)?
            }
            remu_gpu::ComputeIndirectDimensions::Absolute(address) => address,
        };
        let dimensions = agc_read_compute_dimensions(dimensions_address)?;
        (
            remu_gpu::ComputeDispatchSource::Indirect { dimensions_address },
            remu_gpu::decode_compute_launch(dimensions, indirect.initiator, &shader_registers)?,
        )
    };

    let shader_address = remu_gpu::compute_program_address(&shader_registers)
        .ok_or(AgcComputeDispatchError::MissingProgramAddress)?;
    if shader_address == 0 {
        return Err(AgcComputeDispatchError::NullProgramAddress);
    }
    let shader = agc_shader_snapshot(shaders, shader_address)
        .ok_or(AgcComputeDispatchError::ShaderDecode(shader_address))?;
    let compute_system_registers = remu_gpu::decode_compute_system_registers(&shader_registers);
    let shader_evaluation = evaluate_shader_register_snapshot(
        &shader_registers,
        &shader,
        AGC_COMPUTE_USER_DATA_0,
        0,
        Some(compute_system_registers),
    )
    .map_err(|reason| AgcComputeDispatchError::ShaderEvaluation {
        address: shader_address,
        reason,
    })?;
    graphics.emitted_compute_dispatches = graphics.emitted_compute_dispatches.saturating_add(1);
    Ok(remu_gpu::ComputeDispatch {
        sequence: graphics.emitted_compute_dispatches,
        source,
        launch,
        shader_address,
        shader_prefix: agc_shader_prefix_snapshot(shader_address),
        shader: Some(shader),
        shader_evaluation: Some(Arc::new(shader_evaluation)),
        shader_registers,
    })
}

fn agc_prepare_draw_color_targets(
    graphics: &mut AgcGraphicsState,
) -> (Vec<remu_gpu::GpuColorTarget>, bool) {
    let (targets, absent) = match agc_color_target_state(&graphics.context_registers) {
        AgcColorTargetState::Absent => (Vec::new(), true),
        AgcColorTargetState::Programmed(targets) => (targets, false),
        AgcColorTargetState::Malformed { slot } => {
            if graphics.emitted_draws <= 8 {
                eprintln!(
                    "[REMu][AGC] draw={} has incomplete enabled CB_COLOR{slot} state",
                    graphics.emitted_draws
                );
            }
            (Vec::new(), false)
        }
    };
    for target in &targets {
        graphics.known_color_targets.insert(target.address, *target);
    }
    (targets, absent)
}

fn agc_draw_image_access(draw: &remu_gpu::IndexedDraw) -> (bool, bool) {
    [
        draw.export_shader_evaluation.as_deref(),
        draw.pixel_shader_evaluation.as_deref(),
    ]
    .into_iter()
    .flatten()
    .flat_map(|evaluation| &evaluation.image_bindings)
    .fold((false, false), |(reads, writes), binding| {
        let writes_binding =
            binding.opcode.starts_with("ImageStore") || binding.opcode.starts_with("ImageAtomic");
        (reads || !writes_binding, writes || writes_binding)
    })
}

fn agc_submit_or_defer_draw(
    graphics: &mut AgcGraphicsState,
    draw: remu_gpu::IndexedDraw,
    color_target_absent: bool,
) {
    let (reads_image, writes_image) = agc_draw_image_access(&draw);
    if color_target_absent
        && !agc_has_active_depth_state(&graphics.context_registers)
        && reads_image
        && !writes_image
    {
        if draw.sequence <= 8 {
            eprintln!(
                "[REMu][AGC] deferred targetless sampled draw={} until the following VideoOut flip",
                draw.sequence
            );
        }
        graphics.pending_targetless_draw = Some(Box::new(draw));
    } else {
        remu_gpu::emit_event(remu_gpu::GpuEvent::IndexedDraw(Box::new(draw)));
    }
}

fn agc_discard_stale_targetless_draw(graphics: &mut AgcGraphicsState) {
    if let Some(stale) = graphics.pending_targetless_draw.take()
        && stale.sequence <= 8
    {
        eprintln!(
            "[REMu][AGC] discarded unresolved targetless draw={} before the next draw",
            stale.sequence
        );
    }
}

fn agc_trace_draw_summary(draw: &remu_gpu::IndexedDraw) {
    if draw.sequence <= 8 {
        eprintln!(
            "[REMu][AGC] draw={} indexed={} count={} instances={} offset={} index_buffer={:#x}/{} resource={:#x} index_size={} primitive={:#x} es={:#x} ps={:#x} target={:#x}",
            draw.sequence,
            draw.indexed,
            draw.index_count,
            draw.instance_count,
            draw.index_offset,
            draw.index_buffer_address,
            draw.index_buffer_count,
            draw.index_resource_address,
            draw.index_size,
            draw.primitive_type,
            draw.export_shader_address,
            draw.pixel_shader_address,
            draw.color_target_address,
        );
    }
}

fn agc_emit_draw(
    graphics: &mut AgcGraphicsState,
    shaders: &mut AgcShaderRegistry,
    indexed: bool,
    element_offset: u32,
    element_count: u32,
    draw_flags: u32,
) {
    if element_count == 0 {
        return;
    }
    agc_discard_stale_targetless_draw(graphics);
    graphics.draw_index_offset = element_offset;
    graphics.emitted_draws = graphics.emitted_draws.saturating_add(1);

    let export_shader_address = agc_program_address(
        &graphics.shader_registers,
        AGC_SPI_SHADER_PGM_LO_ES,
        AGC_SPI_SHADER_PGM_HI_ES,
    );
    let pixel_shader_address = agc_program_address(
        &graphics.shader_registers,
        AGC_SPI_SHADER_PGM_LO_PS,
        AGC_SPI_SHADER_PGM_HI_PS,
    );
    let (color_targets, color_target_absent) = agc_prepare_draw_color_targets(graphics);
    let color_target_address = color_targets.first().map_or(0, |target| target.address);
    let primitive_type = graphics
        .user_config_registers
        .get(&AGC_VGT_PRIMITIVE_TYPE)
        .copied()
        .unwrap_or_default();
    let export_shader = agc_shader_snapshot(shaders, export_shader_address);
    let pixel_shader = agc_shader_snapshot(shaders, pixel_shader_address);
    let mut draw = remu_gpu::IndexedDraw {
        sequence: graphics.emitted_draws,
        indexed,
        instance_count: graphics.instance_count.max(1),
        index_buffer_address: graphics.index_buffer_address,
        index_resource_address: graphics.index_resource_address,
        index_buffer_count: graphics.index_buffer_count,
        index_size: graphics.index_size,
        index_offset: element_offset,
        index_count: element_count,
        draw_flags,
        primitive_type,
        export_shader_address,
        pixel_shader_address,
        export_shader_prefix: agc_shader_prefix_snapshot(export_shader_address),
        pixel_shader_prefix: agc_shader_prefix_snapshot(pixel_shader_address),
        export_shader,
        pixel_shader,
        export_shader_evaluation: None,
        pixel_shader_evaluation: None,
        color_target_address,
        scanout_target: None,
        context_registers: agc_register_snapshot(&graphics.context_registers),
        shader_registers: agc_register_snapshot(&graphics.shader_registers),
        user_config_registers: agc_register_snapshot(&graphics.user_config_registers),
    };
    let export_user_data_base = export_user_data_register(&draw);
    draw.export_shader_evaluation = draw.export_shader.as_ref().and_then(|snapshot| {
        evaluate_shader_snapshot(&draw, snapshot, export_user_data_base, 8)
            .map(Arc::new)
            .map_err(|error| {
                eprintln!(
                    "[REMu][AGC][SHADER] export evaluation failed at {:#x}: {error}",
                    draw.export_shader_address
                );
            })
            .ok()
    });
    draw.pixel_shader_evaluation = draw.pixel_shader.as_ref().and_then(|snapshot| {
        evaluate_shader_snapshot(&draw, snapshot, 0x00c, 0)
            .map(Arc::new)
            .map_err(|error| {
                eprintln!(
                    "[REMu][AGC][SHADER] pixel evaluation failed at {:#x}: {error}",
                    draw.pixel_shader_address
                );
            })
            .ok()
    });
    agc_trace_draw_summary(&draw);
    if trace_agc_shaders()
        && graphics
            .traced_shader_pairs
            .insert((draw.export_shader_address, draw.pixel_shader_address))
        && graphics.traced_shader_pairs.len() <= 16
    {
        trace_agc_shader_draw(&draw);
    }
    agc_submit_or_defer_draw(graphics, draw, color_target_absent);
}

fn trace_agc_shader_draw(draw: &remu_gpu::IndexedDraw) {
    let instruction_counts = [&draw.export_shader, &draw.pixel_shader].map(|shader| {
        shader
            .as_ref()
            .map_or(0, |shader| shader.program.instructions.len())
    });
    let raw_counts = [&draw.export_shader, &draw.pixel_shader].map(|shader| {
        shader.as_ref().map_or(0, |shader| {
            shader
                .program
                .instructions
                .iter()
                .filter(|instruction| instruction.opcode.known_name().is_none())
                .count()
        })
    });
    let opcode_lists = [&draw.export_shader, &draw.pixel_shader].map(|shader| {
        shader.as_ref().map_or_else(String::new, |shader| {
            shader
                .program
                .instructions
                .iter()
                .map(|instruction| instruction.opcode.to_string())
                .collect::<Vec<_>>()
                .join(",")
        })
    });
    let export_user_data_base = export_user_data_register(draw);
    let evaluation_summaries = [
        trace_shader_evaluation(draw, draw.export_shader.as_ref(), export_user_data_base, 8),
        trace_shader_evaluation(draw, draw.pixel_shader.as_ref(), 0x00c, 0),
    ];
    let msl_summary = trace_msl_pair(draw, export_user_data_base);
    eprintln!(
        "[REMu][AGC][SHADER] es={:#x} ps={:#x} indexed={} count={} instances={} target={:#x} es_ir={}/{} ps_ir={}/{} es_eval={} ps_eval={} msl={} es_ops=[{}] ps_ops=[{}] es_prefix={:x?} ps_prefix={:x?} cx={:x?} sh={:x?} uc={:x?}",
        draw.export_shader_address,
        draw.pixel_shader_address,
        draw.indexed,
        draw.index_count,
        draw.instance_count,
        draw.color_target_address,
        instruction_counts[0],
        raw_counts[0],
        instruction_counts[1],
        raw_counts[1],
        evaluation_summaries[0],
        evaluation_summaries[1],
        msl_summary,
        opcode_lists[0],
        opcode_lists[1],
        draw.export_shader_prefix,
        draw.pixel_shader_prefix,
        draw.context_registers,
        draw.shader_registers,
        draw.user_config_registers,
    );
}

fn export_user_data_register(draw: &remu_gpu::IndexedDraw) -> u32 {
    [0x08c, 0x0cc, 0x04c]
        .into_iter()
        .find(|base| {
            draw.shader_registers
                .iter()
                .any(|register| register.offset == base - 1)
        })
        .unwrap_or(0x0cc)
}

fn evaluate_shader_snapshot(
    draw: &remu_gpu::IndexedDraw,
    snapshot: &remu_gpu::ShaderSnapshot,
    user_data_register: u32,
    scalar_register_base: u32,
) -> Result<remu_shader::ShaderEvaluation, String> {
    evaluate_shader_register_snapshot(
        &draw.shader_registers,
        snapshot,
        user_data_register,
        scalar_register_base,
        None,
    )
}

fn evaluate_shader_register_snapshot(
    shader_registers: &[remu_gpu::GpuRegister],
    snapshot: &remu_gpu::ShaderSnapshot,
    user_data_register: u32,
    scalar_register_base: u32,
    compute_system_registers: Option<remu_shader::ComputeSystemRegisters>,
) -> Result<remu_shader::ShaderEvaluation, String> {
    let state = remu_shader::build_shader_state(
        Arc::clone(&snapshot.program),
        snapshot.metadata.clone(),
        user_data_register,
        scalar_register_base,
        compute_system_registers,
        |offset| {
            shader_registers
                .iter()
                .find(|register| register.offset == offset)
                .map(|register| register.value)
        },
    )
    .map_err(|error| format!("state-error({error})"))?;
    remu_shader::evaluate_shader(
        &ProcessGuestMemory,
        &state,
        remu_shader::EvaluationOptions::default(),
    )
    .map_err(|error| format!("evaluation-error({error})"))
}

fn trace_msl_pair(draw: &remu_gpu::IndexedDraw, export_user_data_base: u32) -> String {
    let Some(export_snapshot) = draw.export_shader.as_ref() else {
        return "missing-es".to_owned();
    };
    let Some(pixel_snapshot) = draw.pixel_shader.as_ref() else {
        return "missing-ps".to_owned();
    };
    let export_evaluation =
        match evaluate_shader_snapshot(draw, export_snapshot, export_user_data_base, 8) {
            Ok(evaluation) => evaluation,
            Err(error) => return format!("es-{error}"),
        };
    let pixel_evaluation = match evaluate_shader_snapshot(draw, pixel_snapshot, 0x00c, 0) {
        Ok(evaluation) => evaluation,
        Err(error) => return format!("ps-{error}"),
    };
    let mut host_location = 0_u32;
    let pixel_outputs = (0..8)
        .filter_map(|guest_slot| {
            let mask =
                (pixel_snapshot.program.pixel_color_export_masks() >> (guest_slot * 4)) & 0xf;
            (mask != 0).then(|| {
                let binding = remu_shader::PixelOutputBinding {
                    guest_slot,
                    host_location,
                    kind: remu_shader::PixelOutputKind::Float,
                };
                host_location += 1;
                binding
            })
        })
        .collect();
    let pixel_options = remu_shader_metal::MetalCompileOptions::pixel(
        pixel_outputs,
        draw.context_registers
            .iter()
            .find(|register| register.offset == 0x1b3)
            .map_or(0, |register| register.value),
        draw.context_registers
            .iter()
            .find(|register| register.offset == 0x1b4)
            .map_or(0, |register| register.value),
    );
    let pixel = match remu_shader_metal::compile_msl(
        &pixel_snapshot.program,
        &pixel_evaluation,
        &pixel_options,
    ) {
        Ok(shader) => shader,
        Err(error) => {
            dump_shader_trace(draw, export_snapshot, pixel_snapshot, None, None);
            return format!("ps-error({error})");
        }
    };
    let vertex_options =
        remu_shader_metal::MetalCompileOptions::vertex(pixel.interpolated_attribute_count);
    let vertex = match remu_shader_metal::compile_msl(
        &export_snapshot.program,
        &export_evaluation,
        &vertex_options,
    ) {
        Ok(shader) => shader,
        Err(error) => {
            dump_shader_trace(draw, export_snapshot, pixel_snapshot, None, Some(&pixel));
            return format!("es-error({error})");
        }
    };
    dump_shader_trace(
        draw,
        export_snapshot,
        pixel_snapshot,
        Some(&vertex),
        Some(&pixel),
    );
    format!(
        "ok(es={}B/{}b/{}t,ps={}B/{}b/{}t)",
        vertex.source.len(),
        vertex.bindings.global_buffer_count,
        vertex.bindings.texture_count,
        pixel.source.len(),
        pixel.bindings.global_buffer_count,
        pixel.bindings.texture_count,
    )
}

fn dump_shader_trace(
    draw: &remu_gpu::IndexedDraw,
    export_snapshot: &remu_gpu::ShaderSnapshot,
    pixel_snapshot: &remu_gpu::ShaderSnapshot,
    vertex: Option<&remu_shader_metal::MetalShader>,
    pixel: Option<&remu_shader_metal::MetalShader>,
) {
    let Some(directory) = std::env::var_os("REMU_DUMP_MSL_DIR").map(PathBuf::from) else {
        return;
    };
    if let Err(error) = std::fs::create_dir_all(&directory) {
        eprintln!("[REMu][AGC][SHADER] could not create MSL dump directory: {error}");
        return;
    }
    let stem = format!(
        "{:016x}-{:016x}",
        draw.export_shader_address, draw.pixel_shader_address
    );
    let ir = format!(
        "export={:#?}\n\npixel={:#?}\n",
        export_snapshot.program.instructions, pixel_snapshot.program.instructions
    );
    for (suffix, contents) in [
        ("ir.txt", Some(ir.as_str())),
        ("vertex.metal", vertex.map(|shader| shader.source.as_str())),
        ("pixel.metal", pixel.map(|shader| shader.source.as_str())),
    ] {
        let Some(contents) = contents else {
            continue;
        };
        if let Err(error) = std::fs::write(directory.join(format!("{stem}-{suffix}")), contents) {
            eprintln!("[REMu][AGC][SHADER] could not write MSL trace: {error}");
        }
    }
}

fn trace_shader_evaluation(
    draw: &remu_gpu::IndexedDraw,
    snapshot: Option<&remu_gpu::ShaderSnapshot>,
    user_data_register: u32,
    scalar_register_base: u32,
) -> String {
    let Some(snapshot) = snapshot else {
        return "missing".to_owned();
    };
    match evaluate_shader_snapshot(draw, snapshot, user_data_register, scalar_register_base) {
        Ok(evaluation) => format!(
            "ok(i={},g={},v={})",
            evaluation.image_bindings.len(),
            evaluation.global_memory_bindings.len(),
            evaluation.vertex_inputs.len(),
        ),
        Err(error) => format!("error({error})"),
    }
}

fn agc_emit_indexed_draw(
    packet_address: u64,
    graphics: &mut AgcGraphicsState,
    shaders: &mut AgcShaderRegistry,
) {
    agc_emit_draw(
        graphics,
        shaders,
        true,
        read_guest_u32(packet_address + 8),
        read_guest_u32(packet_address + 12),
        read_guest_u32(packet_address + 16),
    );
}

#[derive(Clone, Copy)]
struct AgcSubmissionPacket {
    address: u64,
    length: u32,
    opcode: u32,
    register: u32,
}

fn agc_apply_clear_state(
    packet_address: u64,
    length: u32,
    graphics: &mut AgcGraphicsState,
) -> Result<(), AgcSubmissionError> {
    if length != 2 {
        return Err(AgcSubmissionError::ClearStatePacketLength { actual: length });
    }
    let control = read_guest_u32(packet_address + 4);
    if control & !0x0f != 0 {
        return Err(AgcSubmissionError::UnsupportedClearStateControl(control));
    }
    // Gen5's DcbResetQueue emits CLEAR_STATE with a four-bit state selector. KytyPS5's command
    // processor validates that selector and resets the complete context, shader, user-config,
    // index, instance and indirect-argument state regardless of its admitted value. Resource
    // residency and telemetry are host bookkeeping, not register state.
    graphics.reset_pipeline();
    Ok(())
}

fn agc_handle_submission_packet(
    packet: AgcSubmissionPacket,
    graphics: &mut AgcGraphicsState,
    shaders: &mut AgcShaderRegistry,
    index_resources: &BTreeMap<u64, u64>,
    video_out: &VideoOutState,
    flips: &mut Vec<(u32, u32, u64)>,
) -> Result<(), AgcSubmissionError> {
    let AgcSubmissionPacket {
        address: packet_address,
        length,
        opcode,
        register,
    } = packet;
    if opcode == AGC_IT_CLEAR_STATE {
        agc_apply_clear_state(packet_address, length, graphics)?;
    }
    if opcode == AGC_IT_NOP && matches!(register, AGC_R_DRAW_RESET | AGC_R_ACB_RESET) && length >= 2
    {
        graphics.reset_pipeline();
    }
    agc_apply_submission_registers(packet_address, length, opcode, register, graphics)?;
    if opcode == AGC_IT_SET_BASE {
        agc_apply_set_base(packet_address, length, graphics)?;
    }
    if opcode == AGC_IT_NOP && register == AGC_R_ACQUIRE_MEM && length >= 8 {
        remu_gpu::acquire_memory(decode_agc_memory_acquire(packet_address));
    }
    if opcode == AGC_IT_INDEX_BASE && length >= 3 {
        graphics.index_buffer_address = u64::from(read_guest_u32(packet_address + 4))
            | (u64::from(read_guest_u32(packet_address + 8)) << 32);
        graphics.index_resource_address = index_resources
            .get(&graphics.index_buffer_address)
            .copied()
            .unwrap_or_default();
    }
    if opcode == AGC_IT_INDEX_BUFFER_SIZE && length >= 2 {
        graphics.index_buffer_count = read_guest_u32(packet_address + 4);
    }
    if opcode == AGC_IT_INDEX_TYPE && length >= 2 {
        graphics.index_size = read_guest_u32(packet_address + 4) & 0x3;
    }
    if opcode == AGC_IT_NUM_INSTANCES && length >= 2 {
        graphics.instance_count = read_guest_u32(packet_address + 4).max(1);
    }
    if opcode == AGC_IT_DRAW_INDEX_OFFSET && length >= 5 {
        agc_emit_indexed_draw(packet_address, graphics, shaders);
    }
    if opcode == AGC_IT_DRAW_INDEX_AUTO && length >= 3 {
        agc_emit_draw(
            graphics,
            shaders,
            false,
            0,
            read_guest_u32(packet_address + 4),
            read_guest_u32(packet_address + 8),
        );
    }
    if opcode == AGC_IT_NOP && register == AGC_R_DRAW_INDEX_AUTO && length >= 2 {
        agc_emit_draw(
            graphics,
            shaders,
            false,
            0,
            read_guest_u32(packet_address + 4),
            0,
        );
    }
    if matches!(opcode, AGC_IT_DISPATCH_DIRECT | AGC_IT_DISPATCH_INDIRECT) {
        let dispatch =
            agc_decode_compute_dispatch(packet_address, length, opcode, graphics, shaders)
                .map_err(|source| AgcSubmissionError::Compute {
                    packet_address,
                    source,
                })?;
        if dispatch.sequence <= 8 {
            eprintln!(
                "[REMu][AGC] compute={} source={:?} grid={:?} base={:?} local={:?} wave={} cs={:#x}",
                dispatch.sequence,
                dispatch.source,
                dispatch.launch.grid,
                dispatch.launch.base_group,
                dispatch.launch.threads_per_group,
                dispatch.launch.wave_lane_count,
                dispatch.shader_address,
            );
        }
        remu_gpu::emit_event(remu_gpu::GpuEvent::ComputeDispatch(Box::new(dispatch)));
    }
    if opcode == AGC_IT_NOP && register == AGC_R_FLIP && length >= 6 {
        let handle = read_guest_u32(packet_address + 4);
        let buffer_index = read_guest_u32(packet_address + 8);
        let flip_argument = u64::from(read_guest_u32(packet_address + 16))
            | (u64::from(read_guest_u32(packet_address + 20)) << 32);
        if agc_submit_ordered_video_out_flip(
            graphics,
            video_out,
            handle,
            buffer_index,
            flip_argument,
        ) {
            flips.push((handle, buffer_index, flip_argument));
        }
    }
    Ok(())
}

fn decode_agc_memory_acquire(packet_address: u64) -> remu_gpu::GpuMemoryAcquire {
    let coher_control = read_guest_u32(packet_address + 4);
    let size_units = u64::from(read_guest_u32(packet_address + 8))
        | (u64::from(read_guest_u32(packet_address + 12) & 0xff) << 32);
    let base_units = u64::from(read_guest_u32(packet_address + 16))
        | (u64::from(read_guest_u32(packet_address + 20) & 0x00ff_ffff) << 32);
    remu_gpu::GpuMemoryAcquire {
        engine: coher_control >> 31,
        cb_db_control: coher_control & 0x7fff_ffff,
        base_address: base_units << 8,
        size_bytes: size_units << 8,
        poll_interval: read_guest_u32(packet_address + 24) & 0xffff,
        gcr_control: read_guest_u32(packet_address + 28) & 0x7_ffff,
    }
}

struct AgcSubmissionContext<'a> {
    graphics: &'a mut AgcGraphicsState,
    shaders: &'a mut AgcShaderRegistry,
    index_resources: &'a BTreeMap<u64, u64>,
    video_out: &'a VideoOutState,
    submission_count: &'a mut u64,
    flips: &'a mut Vec<(u32, u32, u64)>,
}

fn agc_parse_submission(
    command_address: u64,
    dword_count: u32,
    context: &mut AgcSubmissionContext<'_>,
) -> Result<(), AgcSubmissionError> {
    let mut offset = 0_u32;
    let mut packets = BTreeMap::<(u32, u32), u32>::new();
    while offset < dword_count {
        let packet_address = command_address + u64::from(offset) * 4;
        let header = read_guest_u32(packet_address);
        let packet_type = header >> 30;
        if packet_type == 2 {
            offset += 1;
            continue;
        }
        if packet_type != 3 {
            return Err(AgcSubmissionError::UnsupportedPacketType {
                dword: offset,
                packet_type,
                header,
            });
        }
        let length = ((header >> 16) & 0x3fff) + 2;
        let remaining = dword_count - offset;
        if length > remaining {
            return Err(AgcSubmissionError::InvalidPacketLength {
                dword: offset,
                length,
                remaining,
            });
        }
        let opcode = (header >> 8) & 0xff;
        let register = (header >> 2) & 0x3f;
        *packets.entry((opcode, register)).or_default() += 1;
        agc_handle_submission_packet(
            AgcSubmissionPacket {
                address: packet_address,
                length,
                opcode,
                register,
            },
            context.graphics,
            context.shaders,
            context.index_resources,
            context.video_out,
            context.flips,
        )?;
        offset += length;
    }

    *context.submission_count = context.submission_count.saturating_add(1);
    if *context.submission_count <= 8 {
        let packet_summary = packets
            .iter()
            .map(|(&(opcode, register), &count)| {
                format!("op={opcode:#04x}/reg={register:#04x}:{count}")
            })
            .collect::<Vec<_>>()
            .join(",");
        eprintln!(
            "[REMu][AGC] submit={} address={command_address:#x} dwords={dword_count} parsed={offset} packets=[{packet_summary}] flips={}",
            context.submission_count,
            context.flips.len()
        );
    }
    Ok(())
}

fn agc_driver_submit_dcb(arguments: [u64; 6], state: &mut ProbeDispatchState) -> u64 {
    let packet = arguments[0];
    if packet == 0 {
        return u64::MAX;
    }
    let command_address = read_guest_u64(packet);
    let dword_count = read_guest_u32(packet + 8);
    if command_address == 0 || dword_count == 0 || dword_count > 1_000_000 {
        return u64::MAX;
    }
    let mut flips = Vec::new();
    let result = agc_parse_submission(
        command_address,
        dword_count,
        &mut AgcSubmissionContext {
            graphics: &mut state.agc_graphics,
            shaders: &mut state.agc_shader_registry,
            index_resources: &state.agc_index_resources,
            video_out: &state.video_out,
            submission_count: &mut state.agc_submission_count,
            flips: &mut flips,
        },
    );
    for (handle, buffer_index, flip_argument) in flips {
        queue_video_out_flip(state, handle, buffer_index, flip_argument);
    }
    match result {
        Ok(()) => 0,
        Err(error) => {
            eprintln!("[REMu][AGC] rejected DCB submission: {error}");
            u64::MAX
        }
    }
}

fn agc_driver_submit_multi_dcbs(arguments: [u64; 6], state: &mut ProbeDispatchState) -> u64 {
    let addresses = arguments[0];
    let sizes = arguments[1];
    let count = low_u32(arguments[2]);
    if addresses == 0 || sizes == 0 || count == 0 || count > 4096 {
        return u64::MAX;
    }
    for index in 0..count {
        let command_address = read_guest_u64(addresses + u64::from(index) * 8);
        let dword_count = read_guest_u32(sizes + u64::from(index) * 4);
        if command_address != 0 && dword_count != 0 && dword_count <= 1_000_000 {
            let mut flips = Vec::new();
            let result = agc_parse_submission(
                command_address,
                dword_count,
                &mut AgcSubmissionContext {
                    graphics: &mut state.agc_graphics,
                    shaders: &mut state.agc_shader_registry,
                    index_resources: &state.agc_index_resources,
                    video_out: &state.video_out,
                    submission_count: &mut state.agc_submission_count,
                    flips: &mut flips,
                },
            );
            for (handle, buffer_index, flip_argument) in flips {
                queue_video_out_flip(state, handle, buffer_index, flip_argument);
            }
            match result {
                Ok(()) => {}
                Err(error) => {
                    eprintln!("[REMu][AGC] rejected DCB {index} in multi-submission: {error}");
                    return u64::MAX;
                }
            }
        }
    }
    0
}

fn agc_resolve_acb_descriptor(
    command_address: u64,
    dword_count: u32,
) -> Result<(u64, u32), AgcSubmissionError> {
    const ACB_DESCRIPTOR_MAGIC: u32 = 0x5533_ccaa;
    if dword_count < 5
        || read_guest_u32(command_address + 16) != ACB_DESCRIPTOR_MAGIC
        || read_guest_u32(command_address + 12) != 0
    {
        return Ok((command_address, dword_count));
    }
    let descriptor_address = read_guest_u64(command_address);
    let descriptor_dword_count = read_guest_u32(command_address + 8);
    if descriptor_address == 0 || descriptor_dword_count == 0 || descriptor_dword_count > 1_000_000
    {
        return Err(AgcSubmissionError::InvalidAcbDescriptor {
            address: descriptor_address,
            dword_count: descriptor_dword_count,
        });
    }
    Ok((descriptor_address, descriptor_dword_count))
}

fn agc_submit_compute_commands(
    owner: u32,
    command_address: u64,
    dword_count: u32,
    state: &mut ProbeDispatchState,
    flips: &mut Vec<(u32, u32, u64)>,
) -> Result<(), AgcSubmissionError> {
    let (command_address, dword_count) = agc_resolve_acb_descriptor(command_address, dword_count)?;
    let graphics = state.agc_compute_queues.entry(owner).or_default();
    agc_parse_submission(
        command_address,
        dword_count,
        &mut AgcSubmissionContext {
            graphics,
            shaders: &mut state.agc_shader_registry,
            index_resources: &state.agc_index_resources,
            video_out: &state.video_out,
            submission_count: &mut state.agc_submission_count,
            flips,
        },
    )
}

fn agc_driver_submit_acb(arguments: [u64; 6], state: &mut ProbeDispatchState) -> u64 {
    let owner = low_u32(arguments[0]);
    let packet = arguments[1];
    if packet == 0 {
        return u64::MAX;
    }
    let command_address = read_guest_u64(packet);
    let dword_count = read_guest_u32(packet + 8);
    if command_address == 0 || dword_count == 0 || dword_count > 1_000_000 {
        return u64::MAX;
    }
    let mut flips = Vec::new();
    let result =
        agc_submit_compute_commands(owner, command_address, dword_count, state, &mut flips);
    for (handle, buffer_index, flip_argument) in flips {
        queue_video_out_flip(state, handle, buffer_index, flip_argument);
    }
    match result {
        Ok(()) => 0,
        Err(error) => {
            eprintln!("[REMu][AGC] rejected ACB submission for owner {owner}: {error}");
            u64::MAX
        }
    }
}

fn agc_driver_submit_multi_acbs(arguments: [u64; 6], state: &mut ProbeDispatchState) -> u64 {
    let owner = low_u32(arguments[0]);
    let addresses = arguments[1];
    let sizes = arguments[2];
    let count = low_u32(arguments[3]);
    if addresses == 0 || sizes == 0 || count == 0 || count > 4096 {
        return u64::MAX;
    }
    for index in 0..count {
        let command_address = read_guest_u64(addresses + u64::from(index) * 8);
        let dword_count = read_guest_u32(sizes + u64::from(index) * 4);
        if command_address == 0 || dword_count == 0 || dword_count > 1_000_000 {
            return u64::MAX;
        }
        let mut flips = Vec::new();
        let result =
            agc_submit_compute_commands(owner, command_address, dword_count, state, &mut flips);
        for (handle, buffer_index, flip_argument) in flips {
            queue_video_out_flip(state, handle, buffer_index, flip_argument);
        }
        match result {
            Ok(()) => {}
            Err(error) => {
                eprintln!(
                    "[REMu][AGC] rejected ACB {index} for owner {owner} in multi-submission: {error}"
                );
                return u64::MAX;
            }
        }
    }
    0
}

extern "C" fn guest_locale_virtual_noop(locale: u64) -> u64 {
    locale
}

fn guest_global_locale(state: &mut ProbeDispatchState) -> u64 {
    if state.locale_object != 0 {
        return state.locale_object;
    }

    let vtable = guest_calloc(1, 0x80);
    if vtable == 0 {
        return 0;
    }
    let object = guest_calloc(1, 0x80);
    if object == 0 {
        guest_free(vtable);
        return 0;
    }
    let noop = guest_locale_virtual_noop as *const () as usize as u64;
    for offset in (0_u64..0x60).step_by(8) {
        write_guest_u64(vtable + offset, noop);
    }
    write_guest_u64(object, vtable);
    // Dinkumware's locale implementation begins with a vptr and reference count. The title only
    // invokes the virtual retain/release slots, so a process-lifetime classic locale is sufficient.
    write_guest_u64(object + 8, 1);
    state.locale_vtable = vtable;
    state.locale_object = object;
    object
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

fn guest_tolower_table(state: &mut ProbeDispatchState) -> u64 {
    if state.tolower_table == 0 {
        // Dinkumware's narrow-character case tables contain signed 16-bit values and cover
        // [-128, 255]. The returned pointer addresses the c == 0 element.
        let storage = guest_calloc(384, 2);
        if storage == 0 {
            return 0;
        }
        state.tolower_table = storage + 128 * 2;
        for index in 0_u16..384 {
            let character = i32::from(index) - 128;
            let mapped = if (i32::from(b'A')..=i32::from(b'Z')).contains(&character) {
                character - i32::from(b'A') + i32::from(b'a')
            } else {
                character
            };
            write_guest_u16(
                storage + u64::from(index) * 2,
                i16::try_from(mapped)
                    .expect("case mapping fits i16")
                    .cast_unsigned(),
            );
        }
    }
    state.tolower_table
}

fn guest_toupper_table(state: &mut ProbeDispatchState) -> u64 {
    if state.toupper_table == 0 {
        // Dinkumware indexes the signed-char range as well as 0..=255. Each entry is a 16-bit
        // character value and the exported pointer addresses the c == 0 element.
        let storage = guest_calloc(384, 2);
        if storage == 0 {
            return 0;
        }
        state.toupper_table = storage + 128 * 2;
        for index in 0_u16..384 {
            let character = i32::from(index) - 128;
            let mapped = if (i32::from(b'a')..=i32::from(b'z')).contains(&character) {
                character - i32::from(b'a') + i32::from(b'A')
            } else {
                character
            };
            write_guest_u16(
                storage + u64::from(index) * 2,
                i16::try_from(mapped)
                    .expect("case mapping fits i16")
                    .cast_unsigned(),
            );
        }
    }
    state.toupper_table
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

fn create_pthread_attr(
    address: u64,
    attributes: PthreadAttrState,
    state: &mut ProbeDispatchState,
) -> u64 {
    if address == 0 {
        return sce_kernel_error(SCE_KERNEL_ERROR_INVALID_ARGUMENT);
    }
    if let Some(previous) = state.pthread_attributes.remove(&address) {
        guest_free(previous.handle);
    }
    let handle = guest_calloc(1, 0x40);
    if handle == 0 {
        return sce_kernel_error(SCE_KERNEL_ERROR_TRY_AGAIN);
    }
    write_pthread_attr_object(handle, attributes);
    write_guest_u64(address, handle);
    state
        .pthread_attributes
        .insert(address, PthreadAttrObject { handle, attributes });
    0
}

fn resolve_pthread_attr_key(address: u64, state: &ProbeDispatchState) -> Option<u64> {
    if address == 0 {
        return None;
    }
    if state.pthread_attributes.contains_key(&address) {
        return Some(address);
    }
    state
        .pthread_attributes
        .iter()
        .find_map(|(key, object)| (object.handle == address).then_some(*key))
}

fn pthread_attr_state(address: u64, state: &ProbeDispatchState) -> Option<PthreadAttrState> {
    let key = resolve_pthread_attr_key(address, state)?;
    state
        .pthread_attributes
        .get(&key)
        .map(|object| object.attributes)
}

fn write_pthread_attr_object(address: u64, attributes: PthreadAttrState) {
    write_guest_u64(address, attributes.affinity_mask);
    write_guest_u32(address + 0x08, attributes.detach_state.cast_unsigned());
    write_guest_u64(address + 0x10, attributes.stack_address);
    write_guest_u64(address + 0x18, attributes.stack_size);
    write_guest_u64(address + 0x20, attributes.guard_size);
    write_guest_u32(address + 0x28, attributes.inherit_schedule.cast_unsigned());
    write_guest_u32(address + 0x2c, attributes.schedule_policy.cast_unsigned());
    write_guest_u32(address + 0x30, attributes.schedule_priority.cast_unsigned());
}

fn update_pthread_attr(
    address: u64,
    state: &mut ProbeDispatchState,
    update: impl FnOnce(&mut PthreadAttrState),
) -> u64 {
    let Some(key) = resolve_pthread_attr_key(address, state) else {
        return sce_kernel_error(SCE_KERNEL_ERROR_INVALID_ARGUMENT);
    };
    let object = state
        .pthread_attributes
        .get_mut(&key)
        .expect("the pthread attribute key was just resolved");
    update(&mut object.attributes);
    write_pthread_attr_object(object.handle, object.attributes);
    0
}

fn destroy_pthread_attr(address: u64, state: &mut ProbeDispatchState) -> u64 {
    let Some(key) = resolve_pthread_attr_key(address, state) else {
        return sce_kernel_error(SCE_KERNEL_ERROR_INVALID_ARGUMENT);
    };
    let object = state
        .pthread_attributes
        .remove(&key)
        .expect("the pthread attribute key was just resolved");
    write_guest_u64(key, 0);
    guest_free(object.handle);
    0
}

fn pthread_attr_get(arguments: [u64; 6], state: &mut ProbeDispatchState) -> u64 {
    let thread = arguments[0];
    let output = arguments[1];
    if thread == 0 || output == 0 {
        return sce_kernel_error(SCE_KERNEL_ERROR_INVALID_ARGUMENT);
    }
    let Some(mut attributes) = state.thread_attributes.get(&thread).copied() else {
        return sce_kernel_error(SCE_KERNEL_ERROR_NOT_FOUND);
    };
    if thread == current_guest_thread_handle()
        && let Some((stack_address, stack_size)) = current_native_stack_range()
    {
        attributes.stack_address = stack_address;
        attributes.stack_size = stack_size;
        state.thread_attributes.insert(thread, attributes);
    }
    create_pthread_attr(output, attributes, state)
}

#[cfg(target_os = "macos")]
fn current_native_stack_range() -> Option<(u64, u64)> {
    // SAFETY: pthread_self returns the current live pthread; the two query functions only inspect
    // its immutable stack bounds.
    let thread = unsafe { libc::pthread_self() };
    // SAFETY: `thread` is the current live pthread.
    let top = unsafe { libc::pthread_get_stackaddr_np(thread) } as usize;
    // SAFETY: `thread` is the current live pthread.
    let size = unsafe { libc::pthread_get_stacksize_np(thread) };
    let base = top.checked_sub(size)?;
    Some((u64::try_from(base).ok()?, u64::try_from(size).ok()?))
}

#[cfg(not(target_os = "macos"))]
const fn current_native_stack_range() -> Option<(u64, u64)> {
    None
}

fn pthread_attr_get_stack(arguments: [u64; 6], state: &ProbeDispatchState) -> u64 {
    if arguments[1] == 0 || arguments[2] == 0 {
        return sce_kernel_error(SCE_KERNEL_ERROR_INVALID_ARGUMENT);
    }
    let Some(attributes) = pthread_attr_state(arguments[0], state) else {
        return sce_kernel_error(SCE_KERNEL_ERROR_INVALID_ARGUMENT);
    };
    write_guest_u64(arguments[1], attributes.stack_address);
    write_guest_u64(arguments[2], attributes.stack_size);
    0
}

fn pthread_attr_get_affinity(arguments: [u64; 6], state: &ProbeDispatchState) -> u64 {
    if arguments[1] == 0 {
        return sce_kernel_error(SCE_KERNEL_ERROR_INVALID_ARGUMENT);
    }
    let Some(attributes) = pthread_attr_state(arguments[0], state) else {
        return sce_kernel_error(SCE_KERNEL_ERROR_INVALID_ARGUMENT);
    };
    write_guest_u64(arguments[1], attributes.affinity_mask);
    // The Gen5 kernel ABI returns the observable mask in RAX as well as through the output pointer.
    attributes.affinity_mask
}

fn pthread_attr_set_detach_state(arguments: [u64; 6], state: &mut ProbeDispatchState) -> u64 {
    let detach_state = low_i32(arguments[1]);
    if !matches!(detach_state, 0 | 1) {
        return sce_kernel_error(SCE_KERNEL_ERROR_INVALID_ARGUMENT);
    }
    update_pthread_attr(arguments[0], state, |attributes| {
        attributes.detach_state = detach_state;
    })
}

fn pthread_attr_set_stack_size(arguments: [u64; 6], state: &mut ProbeDispatchState) -> u64 {
    let stack_size = arguments[1];
    if stack_size < MACOS_GUEST_PAGE_ALIGNMENT || usize::try_from(stack_size).is_err() {
        return sce_kernel_error(SCE_KERNEL_ERROR_INVALID_ARGUMENT);
    }
    update_pthread_attr(arguments[0], state, |attributes| {
        attributes.stack_size = stack_size;
        attributes.stack_address = 0;
    })
}

fn pthread_attr_set_schedule_param(arguments: [u64; 6], state: &mut ProbeDispatchState) -> u64 {
    if arguments[1] == 0 {
        return sce_kernel_error(SCE_KERNEL_ERROR_INVALID_ARGUMENT);
    }
    let priority = read_guest_u32(arguments[1]).cast_signed();
    update_pthread_attr(arguments[0], state, |attributes| {
        attributes.schedule_priority = priority;
    })
}

fn create_pthread_mutex_attr(address: u64, state: &mut ProbeDispatchState) -> u64 {
    if address == 0 {
        return sce_kernel_error(SCE_KERNEL_ERROR_INVALID_ARGUMENT);
    }
    if let Some(previous) = state.pthread_mutex_attributes.remove(&address) {
        guest_free(previous.handle);
    }
    let handle = guest_calloc(1, 0x40);
    if handle == 0 {
        return sce_kernel_error(SCE_KERNEL_ERROR_TRY_AGAIN);
    }
    let attributes = PthreadMutexAttrState::default();
    write_pthread_mutex_attr_object(handle, attributes);
    write_guest_u64(address, handle);
    state
        .pthread_mutex_attributes
        .insert(address, PthreadMutexAttrObject { handle, attributes });
    0
}

fn resolve_pthread_mutex_attr_key(address: u64, state: &ProbeDispatchState) -> Option<u64> {
    if address == 0 {
        return None;
    }
    if state.pthread_mutex_attributes.contains_key(&address) {
        return Some(address);
    }
    state
        .pthread_mutex_attributes
        .iter()
        .find_map(|(key, object)| (object.handle == address).then_some(*key))
}

fn pthread_mutex_attr_state(
    address: u64,
    state: &ProbeDispatchState,
) -> Option<PthreadMutexAttrState> {
    let key = resolve_pthread_mutex_attr_key(address, state)?;
    state
        .pthread_mutex_attributes
        .get(&key)
        .map(|object| object.attributes)
}

fn write_pthread_mutex_attr_object(address: u64, attributes: PthreadMutexAttrState) {
    write_guest_u32(address, attributes.mutex_type.cast_unsigned());
    write_guest_u32(address + 4, attributes.protocol.cast_unsigned());
}

fn destroy_pthread_mutex_attr(address: u64, state: &mut ProbeDispatchState) -> u64 {
    let Some(key) = resolve_pthread_mutex_attr_key(address, state) else {
        return sce_kernel_error(SCE_KERNEL_ERROR_INVALID_ARGUMENT);
    };
    let object = state
        .pthread_mutex_attributes
        .remove(&key)
        .expect("the pthread mutex attribute key was just resolved");
    write_guest_u64(key, 0);
    guest_free(object.handle);
    0
}

fn pthread_mutex_attr_set_type(arguments: [u64; 6], state: &mut ProbeDispatchState) -> u64 {
    let mutex_type = match low_i32(arguments[1]) {
        0 | 1 => 1,
        value @ 2..=4 => value,
        _ => return sce_kernel_error(SCE_KERNEL_ERROR_INVALID_ARGUMENT),
    };
    let Some(key) = resolve_pthread_mutex_attr_key(arguments[0], state) else {
        return sce_kernel_error(SCE_KERNEL_ERROR_INVALID_ARGUMENT);
    };
    let object = state
        .pthread_mutex_attributes
        .get_mut(&key)
        .expect("the pthread mutex attribute key was just resolved");
    object.attributes.mutex_type = mutex_type;
    write_pthread_mutex_attr_object(object.handle, object.attributes);
    0
}

fn pthread_condition_attr_init(address: u64, state: &mut ProbeDispatchState) -> u64 {
    if address == 0 {
        return sce_kernel_error(SCE_KERNEL_ERROR_INVALID_ARGUMENT);
    }
    if !state.pthread_condition_attributes.insert(address) {
        return u64::from(libc::EBUSY.cast_unsigned());
    }
    write_guest_u64(address, 0);
    0
}

fn pthread_condition_attr_destroy(address: u64, state: &mut ProbeDispatchState) -> u64 {
    if address == 0 || !state.pthread_condition_attributes.remove(&address) {
        return sce_kernel_error(SCE_KERNEL_ERROR_INVALID_ARGUMENT);
    }
    write_guest_u64(address, 0);
    0
}

fn np_error(result: Result<(), u32>) -> u64 {
    result.map_or_else(sce_kernel_error, |()| 0)
}

fn read_sized_guest_parameters(
    address: u64,
    minimum_size: usize,
    maximum_size: usize,
) -> Option<Vec<u8>> {
    if address == 0 {
        return None;
    }
    let size = usize::try_from(read_guest_u64(address)).ok()?;
    if !(minimum_size..=maximum_size).contains(&size) {
        return None;
    }
    // SAFETY: the ABI's leading size field describes a readable initialization structure owned by
    // the caller. The bounded copy means NP state never retains a guest pointer past this call.
    Some(unsafe { std::slice::from_raw_parts(address as *const u8, size) }.to_vec())
}

fn np_universal_data_initialize(parameter: u64, state: &mut ProbeDispatchState) -> u64 {
    let Some(parameters) = read_sized_guest_parameters(parameter, 16, 0x1000) else {
        return sce_kernel_error(np::UDS_ERROR_INVALID_ARGUMENT);
    };
    np_error(state.np_universal_data.initialize(parameters))
}

fn np_universal_data_terminate(state: &mut ProbeDispatchState) -> u64 {
    np_error(state.np_universal_data.terminate())
}

fn np_universal_data_create_context(arguments: [u64; 6], state: &mut ProbeDispatchState) -> u64 {
    if arguments[0] == 0 {
        return sce_kernel_error(np::UDS_ERROR_INVALID_ARGUMENT);
    }
    match state
        .np_universal_data
        .create_context(low_i32(arguments[1]), arguments[2])
    {
        Ok(context) => write_output_i32(arguments[0], context),
        Err(error) => sce_kernel_error(error),
    }
}

fn np_universal_data_destroy_context(context: u64, state: &mut ProbeDispatchState) -> u64 {
    np_error(state.np_universal_data.destroy_context(low_i32(context)))
}

fn np_universal_data_create_handle(arguments: [u64; 6], state: &mut ProbeDispatchState) -> u64 {
    if arguments[0] == 0 {
        return sce_kernel_error(np::UDS_ERROR_INVALID_ARGUMENT);
    }
    match state.np_universal_data.create_handle() {
        Ok(handle) => write_output_i32(arguments[0], handle),
        Err(error) => sce_kernel_error(error),
    }
}

fn np_universal_data_abort_handle(handle: u64, state: &mut ProbeDispatchState) -> u64 {
    np_error(state.np_universal_data.abort_handle(low_i32(handle)))
}

fn np_universal_data_destroy_handle(handle: u64, state: &mut ProbeDispatchState) -> u64 {
    np_error(state.np_universal_data.destroy_handle(low_i32(handle)))
}

fn np_universal_data_register_context(arguments: [u64; 6], state: &mut ProbeDispatchState) -> u64 {
    np_error(
        state
            .np_universal_data
            .register_context(low_i32(arguments[0]), low_i32(arguments[1])),
    )
}

fn np_universal_data_create_event(arguments: [u64; 6], state: &mut ProbeDispatchState) -> u64 {
    if arguments[2] == 0 || arguments[3] == 0 {
        return sce_kernel_error(np::UDS_ERROR_INVALID_ARGUMENT);
    }
    let Some(name) = read_guest_cstring(arguments[0]) else {
        return sce_kernel_error(np::UDS_ERROR_INVALID_EVENT_NAME);
    };
    match state.np_universal_data.create_event(name, arguments[1]) {
        Ok((event, root_object)) => {
            write_guest_u64(arguments[2], event);
            write_guest_u64(arguments[3], root_object);
            0
        }
        Err(error) => sce_kernel_error(error),
    }
}

fn np_universal_data_destroy_event(event: u64, state: &mut ProbeDispatchState) -> u64 {
    np_error(state.np_universal_data.destroy_event(event))
}

fn np_universal_data_object_set_i32(arguments: [u64; 6], state: &mut ProbeDispatchState) -> u64 {
    let Some(key) = read_guest_cstring(arguments[1]) else {
        return sce_kernel_error(np::UDS_ERROR_INVALID_EVENT_PROPERTY_KEY);
    };
    np_error(
        state
            .np_universal_data
            .object_set_i32(arguments[0], key, low_i32(arguments[2])),
    )
}

fn np_universal_data_object_set_string(arguments: [u64; 6], state: &mut ProbeDispatchState) -> u64 {
    let Some(key) = read_guest_cstring(arguments[1]) else {
        return sce_kernel_error(np::UDS_ERROR_INVALID_EVENT_PROPERTY_KEY);
    };
    let Some(value) = read_guest_cstring(arguments[2]) else {
        return sce_kernel_error(np::UDS_ERROR_INVALID_EVENT_PROPERTY_VALUE);
    };
    np_error(
        state
            .np_universal_data
            .object_set_string(arguments[0], key, value),
    )
}

fn np_universal_data_object_set_array(arguments: [u64; 6], state: &mut ProbeDispatchState) -> u64 {
    if arguments[3] == 0 {
        return sce_kernel_error(np::UDS_ERROR_INVALID_ARGUMENT);
    }
    let Some(key) = read_guest_cstring(arguments[1]) else {
        return sce_kernel_error(np::UDS_ERROR_INVALID_EVENT_PROPERTY_KEY);
    };
    match state
        .np_universal_data
        .object_set_array(arguments[0], key, arguments[2])
    {
        Ok(array) => write_output_u64(arguments[3], array),
        Err(error) => sce_kernel_error(error),
    }
}

fn np_universal_data_array_set_string(arguments: [u64; 6], state: &mut ProbeDispatchState) -> u64 {
    let Some(value) = read_guest_cstring(arguments[1]) else {
        return sce_kernel_error(np::UDS_ERROR_INVALID_EVENT_PROPERTY_VALUE);
    };
    np_error(
        state
            .np_universal_data
            .array_append_string(arguments[0], value),
    )
}

fn np_universal_data_post_event(arguments: [u64; 6], state: &mut ProbeDispatchState) -> u64 {
    match state.np_universal_data.post_event(
        low_i32(arguments[0]),
        low_i32(arguments[1]),
        arguments[2],
        arguments[3],
    ) {
        Ok(posted) => {
            eprintln!(
                "[REMu][NP] committed universal-data event sequence={} context={} user={:#x} name={:?} bytes={} options={:#x}/{:#x}/{:#x}",
                posted.sequence,
                posted.context,
                posted.user_id,
                posted.event_name,
                posted.payload_size,
                posted.context_option,
                posted.event_option,
                posted.post_option,
            );
            0
        }
        Err(error) => sce_kernel_error(error),
    }
}

fn np_game_intent_initialize(parameter: u64, state: &mut ProbeDispatchState) -> u64 {
    let Some(parameters) = read_sized_guest_parameters(parameter, 0x28, 0x1000) else {
        return sce_kernel_error(np::GAME_INTENT_ERROR_INVALID_ARGUMENT);
    };
    np_error(state.np_game_intent.initialize(parameters))
}

fn np_game_intent_receive(output: u64, state: &mut ProbeDispatchState) -> u64 {
    const GAME_INTENT_OUTPUT_SIZE: usize = 0x4140;
    const INTENT_TYPE_OFFSET: u64 = 0x0c;
    const INTENT_TYPE_CAPACITY: usize = 0x100;
    const PROPERTY_OBJECT_OFFSET: u64 = 0x134;
    if output == 0 {
        return sce_kernel_error(np::GAME_INTENT_ERROR_INVALID_ARGUMENT);
    }
    let Ok(output_size) = usize::try_from(read_guest_u64(output)) else {
        return sce_kernel_error(np::GAME_INTENT_ERROR_INVALID_ARGUMENT);
    };
    if output_size < GAME_INTENT_OUTPUT_SIZE {
        return sce_kernel_error(np::GAME_INTENT_ERROR_INSUFFICIENT_BUFFER);
    }
    let intent = match state.np_game_intent.receive() {
        Ok(intent) => intent,
        Err(error) => return sce_kernel_error(error),
    };
    if intent.intent_type.is_empty()
        || intent.intent_type.len() >= INTENT_TYPE_CAPACITY
        || intent.intent_type.as_bytes().contains(&0)
    {
        return sce_kernel_error(np::GAME_INTENT_ERROR_INVALID_ARGUMENT);
    }
    guest_memset(
        output + 8,
        0,
        u64::try_from(GAME_INTENT_OUTPUT_SIZE - 8).expect("fixed intent size fits u64"),
    );
    write_guest_u32(output + 8, 0);
    for (index, byte) in intent.intent_type.bytes().enumerate() {
        write_guest_u8(
            output + INTENT_TYPE_OFFSET + u64::try_from(index).expect("intent type index fits u64"),
            byte,
        );
    }
    let property_object = output + PROPERTY_OBJECT_OFFSET;
    state
        .np_game_intent
        .retain_received_properties(property_object, intent.properties);
    0
}

fn np_game_intent_get_property_string(arguments: [u64; 6], state: &mut ProbeDispatchState) -> u64 {
    if arguments[2] == 0 || arguments[3] == 0 {
        return sce_kernel_error(np::GAME_INTENT_ERROR_INVALID_ARGUMENT);
    }
    let Some(key) = read_guest_cstring(arguments[1]) else {
        return sce_kernel_error(np::GAME_INTENT_ERROR_INVALID_ARGUMENT);
    };
    let value = match state.np_game_intent.property(arguments[0], &key) {
        Ok(value) => value.to_owned(),
        Err(error) => return sce_kernel_error(error),
    };
    let Some(required) = value.len().checked_add(1) else {
        return sce_kernel_error(np::GAME_INTENT_ERROR_INSUFFICIENT_BUFFER);
    };
    if u64::try_from(required).unwrap_or(u64::MAX) > arguments[3] {
        return sce_kernel_error(np::GAME_INTENT_ERROR_INSUFFICIENT_BUFFER);
    }
    for (index, byte) in value.bytes().chain(std::iter::once(0)).enumerate() {
        write_guest_u8(
            arguments[2] + u64::try_from(index).expect("property string index fits u64"),
            byte,
        );
    }
    0
}

fn np_trophy_create_context(arguments: [u64; 6], state: &mut ProbeDispatchState) -> u64 {
    if arguments[0] == 0 {
        return sce_kernel_error(np::TROPHY2_ERROR_INVALID_ARGUMENT);
    }
    let context = state
        .np_trophy
        .create_context(low_i32(arguments[1]), arguments[2]);
    write_output_i32(arguments[0], context)
}

fn np_trophy_destroy_context(context: u64, state: &mut ProbeDispatchState) -> u64 {
    np_error(state.np_trophy.destroy_context(low_i32(context)))
}

fn np_trophy_create_handle(arguments: [u64; 6], state: &mut ProbeDispatchState) -> u64 {
    if arguments[0] == 0 {
        return sce_kernel_error(np::TROPHY2_ERROR_INVALID_ARGUMENT);
    }
    let handle = state.np_trophy.create_handle();
    write_output_i32(arguments[0], handle)
}

fn np_trophy_destroy_handle(handle: u64, state: &mut ProbeDispatchState) -> u64 {
    np_error(state.np_trophy.destroy_handle(low_i32(handle)))
}

fn np_trophy_abort_handle(handle: u64, state: &mut ProbeDispatchState) -> u64 {
    np_error(state.np_trophy.abort_handle(low_i32(handle)))
}

fn np_trophy_register_context(arguments: [u64; 6], state: &mut ProbeDispatchState) -> u64 {
    np_error(
        state
            .np_trophy
            .register_context(low_i32(arguments[0]), low_i32(arguments[1])),
    )
}

fn np_trophy_register_unlock_callback(arguments: [u64; 6], state: &mut ProbeDispatchState) -> u64 {
    np_error(
        state
            .np_trophy
            .register_unlock_callback(arguments[0], arguments[1]),
    )
}

fn np_trophy_unregister_unlock_callback(state: &mut ProbeDispatchState) -> u64 {
    np_error(state.np_trophy.unregister_unlock_callback())
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

fn guest_cxa_allocate_exception(size: u64) -> u64 {
    let Some(allocation_size) = size.checked_add(0x80) else {
        return 0;
    };
    let allocation = guest_calloc(1, allocation_size);
    if allocation == 0 {
        0
    } else {
        allocation.checked_add(0x80).unwrap_or(0)
    }
}

fn guest_cxa_free_exception(exception: u64) -> u64 {
    if let Some(allocation) = exception.checked_sub(0x80) {
        guest_free(allocation);
    }
    0
}

fn guest_cxa_throw(arguments: [u64; 6]) -> u64 {
    let exception = arguments[0];
    let type_info = arguments[1];
    eprintln!(
        "[REMu][CXX] guest exception object={exception:#x} type_info={type_info:#x} destructor={:#x} object_word0={:#x} object_word1={:#x}",
        arguments[2],
        read_guest_u64(exception),
        read_guest_u64(exception + 8),
    );
    terminate_probe(88);
}

fn guest_errno_address() -> u64 {
    let thread_pointer = current_guest_thread_pointer();
    if thread_pointer == 0 {
        0
    } else {
        thread_pointer.checked_add(0x40).unwrap_or(0)
    }
}

fn dispatch_guest_numeric_parser(nid: &str, arguments: [u64; 6]) -> ProbeImportReturn {
    if arguments[0] == 0 {
        return ProbeImportReturn::integer(0);
    }
    set_host_errno(0);
    if guest_errno_address() != 0 {
        write_guest_u32(guest_errno_address(), 0);
    }
    let end_pointer = arguments[1] as *mut *mut libc::c_char;
    let result = match nid {
        // Dinkumware _Stoul uses the target's unsigned-long parser and the same three-argument
        // signature as strtoul. Gen5 and the macOS x86-64 host both use a 64-bit unsigned long.
        "zlfEH8FmyUA" => {
            // SAFETY: The guest supplies a NUL-terminated string and optional writable end pointer.
            ProbeImportReturn::integer(unsafe {
                libc::strtoul(
                    arguments[0] as *const libc::c_char,
                    end_pointer,
                    low_i32(arguments[2]),
                )
            })
        }
        // libc strtoull
        "5OqszGpy7Mg" => {
            // SAFETY: The guest supplies a NUL-terminated string and optional writable end pointer.
            ProbeImportReturn::integer(unsafe {
                libc::strtoull(
                    arguments[0] as *const libc::c_char,
                    end_pointer,
                    low_i32(arguments[2]),
                )
            })
        }
        // libc strtoll
        "VOBg+iNwB-4" => {
            // SAFETY: The guest supplies a NUL-terminated string and optional writable end pointer.
            ProbeImportReturn::integer(unsafe {
                libc::strtoll(
                    arguments[0] as *const libc::c_char,
                    end_pointer,
                    low_i32(arguments[2]),
                )
                .cast_unsigned()
            })
        }
        // libc strtod. ProbeImportReturn's mixed INTEGER/SSE ABI returns this through XMM0 while
        // preserving RAX for every ordinary import.
        "2vDqwBlpF-o" => {
            // SAFETY: The guest supplies a NUL-terminated string and optional writable end pointer.
            ProbeImportReturn::float(unsafe {
                libc::strtod(arguments[0] as *const libc::c_char, end_pointer)
            })
        }
        _ => ProbeImportReturn::integer(0),
    };
    let guest_errno = guest_errno_address();
    if guest_errno != 0 {
        write_guest_u32(guest_errno, host_errno().cast_unsigned());
    }
    result
}

#[cfg(target_os = "macos")]
fn set_host_errno(value: i32) {
    // SAFETY: __error returns this host thread's writable errno cell.
    unsafe {
        *libc::__error() = value;
    }
}

#[cfg(target_os = "macos")]
fn host_errno() -> i32 {
    // SAFETY: __error returns this host thread's live errno cell.
    unsafe { *libc::__error() }
}

#[cfg(target_os = "linux")]
fn set_host_errno(value: i32) {
    // SAFETY: __errno_location returns this host thread's writable errno cell.
    unsafe {
        *libc::__errno_location() = value;
    }
}

#[cfg(target_os = "linux")]
fn host_errno() -> i32 {
    // SAFETY: __errno_location returns this host thread's live errno cell.
    unsafe { *libc::__errno_location() }
}

#[cfg(not(any(target_os = "linux", target_os = "macos")))]
fn set_host_errno(_value: i32) {}

#[cfg(not(any(target_os = "linux", target_os = "macos")))]
fn host_errno() -> i32 {
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

#[cfg(test)]
fn read_guest_u16(address: u64) -> u16 {
    // SAFETY: HLE calls use guest pointers into the process's live mapped address space.
    unsafe { std::ptr::read_unaligned(address as *const u16) }
}

fn read_guest_u32(address: u64) -> u32 {
    // SAFETY: HLE calls use guest pointers into the process's live mapped address space.
    unsafe { std::ptr::read_unaligned(address as *const u32) }
}

fn read_guest_u64(address: u64) -> u64 {
    // SAFETY: HLE calls use guest pointers into the process's live mapped address space.
    unsafe { std::ptr::read_unaligned(address as *const u64) }
}

fn write_guest_u8(address: u64, value: u8) {
    // SAFETY: HLE calls use guest pointers into the process's live mapped address space.
    unsafe {
        std::ptr::write_unaligned(address as *mut u8, value);
    }
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
fn allocate_direct_memory(arguments: [u64; 6], state: &mut ProbeDispatchState) -> u64 {
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
fn allocate_direct_memory(_arguments: [u64; 6], _state: &mut ProbeDispatchState) -> u64 {
    sce_kernel_error(SCE_KERNEL_ERROR_NOT_FOUND)
}

#[cfg(target_os = "macos")]
fn map_direct_memory(arguments: [u64; 6], state: &mut ProbeDispatchState) -> u64 {
    map_direct_memory_core(arguments, None, state)
}

#[cfg(target_os = "macos")]
fn map_direct_memory_2(arguments: [u64; 6], alignment: u64, state: &mut ProbeDispatchState) -> u64 {
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
    let result = if let Some(memory_type) = memory_type {
        memory.map_direct_with_memory_type(
            placement,
            length,
            alignment,
            protection,
            direct_start,
            Some(memory_type),
        )
    } else {
        memory.map_direct(placement, length, alignment, protection, direct_start)
    };
    let mapped = match result {
        Ok(mapped) => mapped,
        Err(error) => return kernel_memory_error(&error),
    };
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
fn map_direct_memory(_arguments: [u64; 6], _state: &mut ProbeDispatchState) -> u64 {
    sce_kernel_error(SCE_KERNEL_ERROR_NOT_FOUND)
}

#[cfg(not(target_os = "macos"))]
fn map_direct_memory_2(
    _arguments: [u64; 6],
    _alignment: u64,
    _state: &mut ProbeDispatchState,
) -> u64 {
    sce_kernel_error(SCE_KERNEL_ERROR_NOT_FOUND)
}

#[cfg(target_os = "macos")]
fn kernel_mprotect(arguments: [u64; 6], state: &mut ProbeDispatchState) -> u64 {
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
    let memory = match kernel_memory_mut(state) {
        Ok(memory) => memory,
        Err(error) => return kernel_memory_error(&error),
    };
    if let Err(error) = memory.protect(
        GuestAddress::new(aligned_address),
        aligned_length,
        protection,
    ) {
        return kernel_memory_error(&error);
    }
    if trace_guest_vm() {
        eprintln!(
            "[REMu][VM] protected address={aligned_address:#x} length={aligned_length:#x} protection={protection:#x}"
        );
    }
    0
}

#[cfg(not(target_os = "macos"))]
fn kernel_mprotect(_arguments: [u64; 6], _state: &mut ProbeDispatchState) -> u64 {
    sce_kernel_error(SCE_KERNEL_ERROR_NOT_FOUND)
}

#[cfg(target_os = "macos")]
fn kernel_virtual_query(arguments: [u64; 6], state: &ProbeDispatchState) -> u64 {
    const VIRTUAL_QUERY_INFO_SIZE: u64 = 72;
    let (query, flags, output, output_size) = (
        arguments[0],
        low_i32(arguments[1]),
        arguments[2],
        arguments[3],
    );
    if output == 0 || output_size < VIRTUAL_QUERY_INFO_SIZE {
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
    // Direct and currently mapped, matching the Gen5 SceKernelVirtualQueryInfo flags.
    write_guest_u8(output + 0x20, 0x12);
    for (index, byte) in b"direct".iter().copied().enumerate() {
        write_guest_u8(output + 0x21 + u64::try_from(index).unwrap_or(0), byte);
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
fn kernel_virtual_query(_arguments: [u64; 6], _state: &ProbeDispatchState) -> u64 {
    sce_kernel_error(SCE_KERNEL_ERROR_NOT_FOUND)
}

#[cfg(target_os = "macos")]
fn kernel_munmap(arguments: [u64; 6], state: &mut ProbeDispatchState) -> u64 {
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
    if trace_guest_vm() {
        eprintln!("[REMu][VM] unmapped address={address:#x} length={length:#x}");
    }
    0
}

#[cfg(not(target_os = "macos"))]
fn kernel_munmap(_arguments: [u64; 6], _state: &mut ProbeDispatchState) -> u64 {
    sce_kernel_error(SCE_KERNEL_ERROR_NOT_FOUND)
}

#[cfg(target_os = "macos")]
fn kernel_release_direct_memory(arguments: [u64; 6], state: &mut ProbeDispatchState) -> u64 {
    kernel_release_direct_memory_core(arguments, false, state)
}

#[cfg(target_os = "macos")]
fn kernel_checked_release_direct_memory(
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
            Ok(released) => released,
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
fn kernel_release_direct_memory(_arguments: [u64; 6], _state: &mut ProbeDispatchState) -> u64 {
    sce_kernel_error(SCE_KERNEL_ERROR_NOT_FOUND)
}

#[cfg(not(target_os = "macos"))]
fn kernel_checked_release_direct_memory(
    _arguments: [u64; 6],
    _state: &mut ProbeDispatchState,
) -> u64 {
    sce_kernel_error(SCE_KERNEL_ERROR_NOT_FOUND)
}

fn trace_guest_vm() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| std::env::var_os("REMU_TRACE_VM").is_some_and(|value| value == "1"))
}

fn trace_agc_shaders() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| {
        std::env::var_os("REMU_TRACE_AGC_SHADERS").is_some_and(|value| value == "1")
    })
}

fn trace_guest_time() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| std::env::var_os("REMU_TRACE_TIME").is_some_and(|value| value == "1"))
}

fn trace_guest_sync() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| std::env::var_os("REMU_TRACE_SYNC").is_some_and(|value| value == "1"))
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

fn sanitizer_new_replace_external() -> u64 {
    const TLS_NEW_REPLACE_OFFSET: u64 = 0x300;
    const NEW_REPLACE_SIZE: usize = 0x68;
    let thread_pointer = current_guest_thread_pointer();
    if thread_pointer == 0 {
        return 0;
    }
    let address = thread_pointer + TLS_NEW_REPLACE_OFFSET;
    // SAFETY: every guest thread reserves a writable TCB region after its thread pointer, and this
    // kernel-owned compatibility structure fits entirely within that reservation.
    unsafe {
        std::ptr::write_bytes(address as *mut u8, 0, NEW_REPLACE_SIZE);
    }
    write_guest_u64(address, NEW_REPLACE_SIZE as u64);
    address
}

fn libc_heap_get_trace_info(info_address: u64, state: &mut ProbeDispatchState) -> u64 {
    const HEAP_TRACE_INFO_SIZE: u64 = 32;
    const HEAP_TRACE_STORAGE_WORDS: usize = 65;
    if info_address == 0 || read_guest_u64(info_address) != HEAP_TRACE_INFO_SIZE {
        return sce_kernel_error(SCE_KERNEL_ERROR_INVALID_ARGUMENT);
    }
    if state.heap_trace_storage.len() != HEAP_TRACE_STORAGE_WORDS {
        state.heap_trace_storage.resize(HEAP_TRACE_STORAGE_WORDS, 0);
    }
    let storage_address = state.heap_trace_storage.as_mut_ptr() as usize as u64;
    write_guest_u64(info_address + 16, storage_address);
    write_guest_u64(info_address + 24, storage_address + 8);
    0
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

fn guest_strstr(haystack: u64, needle: u64) -> u64 {
    if haystack == 0 || needle == 0 {
        return 0;
    }
    // SAFETY: strstr's guest ABI contract supplies two NUL-terminated strings. Its result points
    // into the unchanged haystack, which is already process-addressable guest memory.
    unsafe {
        libc::strstr(
            haystack as *const libc::c_char,
            needle as *const libc::c_char,
        ) as usize as u64
    }
}

#[cfg(target_arch = "x86_64")]
fn save_data_root(game_root: &Path) -> PathBuf {
    let base = std::env::var_os("REMU_SAVEDATA_DIR")
        .map_or_else(|| std::env::temp_dir().join("remu-savedata"), PathBuf::from);
    let title = game_root
        .file_name()
        .and_then(|name| name.to_str())
        .and_then(|name| name.strip_suffix("-app0").or(Some(name)))
        .unwrap_or("unknown-title");
    base.join(sanitize_save_segment(title))
}

fn sanitize_save_segment(value: &str) -> String {
    let value = value
        .chars()
        .map(|character| {
            if character.is_ascii_alphanumeric() || matches!(character, '-' | '_') {
                character
            } else {
                '_'
            }
        })
        .collect::<String>();
    if value.trim_matches('_').is_empty() {
        "default".to_owned()
    } else {
        value
    }
}

const SAVE_DATA_ERROR_PARAMETER: u32 = 0x809f_0000;
const SAVE_DATA_ERROR_NOT_INITIALIZED: u32 = 0x809f_0001;
const SAVE_DATA_ERROR_OUT_OF_MEMORY: u32 = 0x809f_0002;
const SAVE_DATA_ERROR_BUSY: u32 = 0x809f_0003;
const SAVE_DATA_ERROR_NOT_MOUNTED: u32 = 0x809f_0004;
const SAVE_DATA_ERROR_EXISTS: u32 = 0x809f_0007;
const SAVE_DATA_ERROR_NOT_FOUND: u32 = 0x809f_0008;
const SAVE_DATA_ERROR_INTERNAL: u32 = 0x809f_000b;
const SAVE_DATA_ERROR_MOUNT_FULL: u32 = 0x809f_000c;
const SAVE_DATA_ERROR_BAD_MOUNTED: u32 = 0x809f_000d;
const SAVE_DATA_DIR_NAME_SIZE: usize = 32;
const SAVE_DATA_PARAM_SIZE: usize = 0x530;
const SAVE_DATA_SEARCH_INFO_SIZE: usize = 0x30;
const SAVE_DATA_BLOCK_SIZE: u64 = 32 * 1024;
const SAVE_DATA_MIN_BLOCKS: u64 = 96;
const SAVE_DATA_MAX_ICON_SIZE: u64 = 16 * 1024 * 1024;
const SAVE_DATA_MOUNT_MODE_READ_ONLY: u32 = 1 << 0;
const SAVE_DATA_MOUNT_MODE_READ_WRITE: u32 = 1 << 1;
const SAVE_DATA_MOUNT_MODE_CREATE: u32 = 1 << 2;
const SAVE_DATA_MOUNT_MODE_CREATE_IF_MISSING: u32 = 1 << 5;

#[derive(Debug)]
struct SaveDataSearchEntry {
    name: String,
    param: [u8; SAVE_DATA_PARAM_SIZE],
    blocks: u64,
    free_blocks: u64,
    modified_seconds: i64,
    user_param: u32,
}

fn save_data_initialize(state: &mut ProbeDispatchState) -> u64 {
    match std::fs::create_dir_all(&state.save_root) {
        Ok(()) => {
            state.save_data_initialized = true;
            eprintln!(
                "[REMu][SAVEDATA] initialized root={:?}",
                state.save_root.display()
            );
            0
        }
        Err(error) => {
            eprintln!("[REMu][SAVEDATA] initialize failed: {error}");
            sce_kernel_error(SAVE_DATA_ERROR_INTERNAL)
        }
    }
}

fn save_data_dir_name_search(arguments: [u64; 6], state: &ProbeDispatchState) -> u64 {
    if !state.save_data_initialized {
        return sce_kernel_error(SAVE_DATA_ERROR_NOT_INITIALIZED);
    }
    let (condition, result) = (arguments[0], arguments[1]);
    if condition == 0 || result == 0 {
        return sce_kernel_error(SAVE_DATA_ERROR_PARAMETER);
    }

    let user_id = low_i32(u64::from(read_guest_u32(condition)));
    let title_id_address = read_guest_u64(condition + 0x08);
    let pattern_address = read_guest_u64(condition + 0x10);
    let sort_key = read_guest_u32(condition + 0x18);
    let sort_order = read_guest_u32(condition + 0x1c);
    if user_id < 0 || !matches!(sort_key, 0 | 1 | 2 | 3 | 5) || sort_order > 1 {
        return sce_kernel_error(SAVE_DATA_ERROR_PARAMETER);
    }
    let pattern = if pattern_address == 0 {
        String::new()
    } else {
        let Some(pattern) = read_guest_fixed_ascii(pattern_address, SAVE_DATA_DIR_NAME_SIZE) else {
            return sce_kernel_error(SAVE_DATA_ERROR_PARAMETER);
        };
        pattern
    };
    let title_root = if title_id_address == 0 {
        state.save_root.clone()
    } else {
        let Some(title_id) = read_guest_fixed_ascii(title_id_address, 10) else {
            return sce_kernel_error(SAVE_DATA_ERROR_PARAMETER);
        };
        if title_id.is_empty() {
            return sce_kernel_error(SAVE_DATA_ERROR_PARAMETER);
        }
        state
            .save_root
            .parent()
            .unwrap_or(&state.save_root)
            .join(sanitize_save_segment(&title_id))
    };

    let mut entries = match enumerate_save_data_entries(&title_root, &pattern) {
        Ok(entries) => entries,
        Err(error) => {
            eprintln!(
                "[REMu][SAVEDATA] search failed root={:?}: {error}",
                title_root.display()
            );
            return sce_kernel_error(SAVE_DATA_ERROR_INTERNAL);
        }
    };
    sort_save_data_entries(&mut entries, sort_key, sort_order);

    let dir_names_address = read_guest_u64(result + 0x08);
    let dir_names_capacity = read_guest_u32(result + 0x10) as usize;
    let params_address = read_guest_u64(result + 0x18);
    let infos_address = read_guest_u64(result + 0x20);
    let set_count = dir_names_capacity.min(entries.len());
    if set_count != 0 && dir_names_address == 0 {
        return sce_kernel_error(SAVE_DATA_ERROR_PARAMETER);
    }
    write_guest_u32(result, u32::try_from(entries.len()).unwrap_or(u32::MAX));
    write_guest_u32(
        result + 0x14,
        u32::try_from(set_count).expect("set count is bounded by a u32 capacity"),
    );
    for (index, entry) in entries.iter().take(set_count).enumerate() {
        let name_address = dir_names_address
            + u64::try_from(index * SAVE_DATA_DIR_NAME_SIZE).expect("save result offset fits u64");
        write_guest_fixed_ascii(name_address, SAVE_DATA_DIR_NAME_SIZE, &entry.name);
        if params_address != 0 {
            let address = params_address
                + u64::try_from(index * SAVE_DATA_PARAM_SIZE).expect("param offset fits u64");
            write_guest_bytes(address, &entry.param);
        }
        if infos_address != 0 {
            let address = infos_address
                + u64::try_from(index * SAVE_DATA_SEARCH_INFO_SIZE)
                    .expect("search info offset fits u64");
            // SAFETY: the ABI supplies one writable 0x30-byte info record per requested result.
            unsafe { std::ptr::write_bytes(address as *mut u8, 0, SAVE_DATA_SEARCH_INFO_SIZE) };
            write_guest_u64(address, entry.blocks);
            write_guest_u64(address + 0x08, entry.free_blocks);
        }
    }
    eprintln!(
        "[REMu][SAVEDATA] search user={user_id} pattern={pattern:?} sort={sort_key}/{sort_order} hits={} set={set_count}",
        entries.len()
    );
    0
}

fn save_data_create_transaction_resource(
    arguments: [u64; 6],
    state: &mut ProbeDispatchState,
) -> u64 {
    if !state.save_data_initialized {
        return sce_kernel_error(SAVE_DATA_ERROR_NOT_INITIALIZED);
    }
    if arguments[0] == 0 {
        return sce_kernel_error(SAVE_DATA_ERROR_PARAMETER);
    }
    let id = state.next_save_transaction_resource.max(1);
    let Some(next) = id.checked_add(1) else {
        return sce_kernel_error(SAVE_DATA_ERROR_OUT_OF_MEMORY);
    };
    state.next_save_transaction_resource = next;
    state.save_transaction_resources.insert(id);
    eprintln!(
        "[REMu][SAVEDATA] transaction resource={id} memory_size={:#x}",
        arguments[0]
    );
    u64::from(id)
}

fn save_data_mount3(arguments: [u64; 6], state: &mut ProbeDispatchState) -> u64 {
    const MOUNT_RESULT_SIZE: usize = 0x40;
    if !state.save_data_initialized {
        return sce_kernel_error(SAVE_DATA_ERROR_NOT_INITIALIZED);
    }
    let (mount, result) = (arguments[0], arguments[1]);
    if mount == 0 || result == 0 {
        return sce_kernel_error(SAVE_DATA_ERROR_PARAMETER);
    }
    let user_id = low_i32(u64::from(read_guest_u32(mount)));
    let dir_name_address = read_guest_u64(mount + 0x08);
    let blocks = read_guest_u64(mount + 0x10);
    let system_blocks = read_guest_u64(mount + 0x18);
    let mount_mode = read_guest_u32(mount + 0x20);
    let mode = read_guest_u32(mount + 0x24);
    let resource = read_guest_u32(mount + 0x28);
    let Some(dir_name) = read_guest_fixed_ascii(dir_name_address, SAVE_DATA_DIR_NAME_SIZE) else {
        return sce_kernel_error(SAVE_DATA_ERROR_PARAMETER);
    };
    if user_id < 0 || dir_name.is_empty() {
        return sce_kernel_error(SAVE_DATA_ERROR_PARAMETER);
    }
    if resource != 0 && !state.save_transaction_resources.contains(&resource) {
        return sce_kernel_error(SAVE_DATA_ERROR_PARAMETER);
    }

    let slot = state.save_root.join(sanitize_save_segment(&dir_name));
    let existed = slot.is_dir();
    let create = mount_mode & SAVE_DATA_MOUNT_MODE_CREATE != 0;
    let create_if_missing = mount_mode & SAVE_DATA_MOUNT_MODE_CREATE_IF_MISSING != 0;
    if !existed && !create && !create_if_missing {
        return sce_kernel_error(SAVE_DATA_ERROR_NOT_FOUND);
    }
    if existed && create {
        return sce_kernel_error(SAVE_DATA_ERROR_EXISTS);
    }
    if state
        .save_mounts
        .values()
        .any(|mounted| mounted.path == slot)
    {
        return sce_kernel_error(SAVE_DATA_ERROR_BUSY);
    }
    if !existed && let Err(error) = create_save_data_slot(&slot, &dir_name, blocks) {
        eprintln!(
            "[REMu][SAVEDATA] mount creation failed path={:?}: {error}",
            slot.display()
        );
        return sce_kernel_error(SAVE_DATA_ERROR_INTERNAL);
    }

    let Some(mount_index) =
        (0_u8..16).find(|index| !state.save_mounts.contains_key(&format!("/savedata{index}")))
    else {
        return sce_kernel_error(SAVE_DATA_ERROR_MOUNT_FULL);
    };
    let mount_point = format!("/savedata{mount_index}");

    // SAFETY: sceSaveDataMount3 supplies a writable 0x40-byte result structure.
    unsafe {
        std::ptr::write_bytes(result as *mut u8, 0, MOUNT_RESULT_SIZE);
    }
    write_guest_fixed_ascii(result, 16, &mount_point);
    write_guest_u32(result + 0x1c, u32::from(create_if_missing && !existed));
    state.save_mounts.insert(
        mount_point.clone(),
        SaveDataMount {
            path: slot.clone(),
            committed_path: slot.clone(),
            dir_name: dir_name.clone(),
            user_id,
            resource,
            read_only: mount_mode & SAVE_DATA_MOUNT_MODE_READ_ONLY != 0
                && mount_mode & SAVE_DATA_MOUNT_MODE_READ_WRITE == 0,
        },
    );
    eprintln!(
        "[REMu][SAVEDATA] mount3 user={user_id} dir={dir_name:?} blocks={blocks:#x} system_blocks={system_blocks:#x} mount_mode={mount_mode:#x} resource={resource} mode={mode:#x} mount={mount_point:?} host={:?}",
        slot.display()
    );
    0
}

fn save_data_umount2(arguments: [u64; 6], state: &mut ProbeDispatchState) -> u64 {
    if !state.save_data_initialized {
        return sce_kernel_error(SAVE_DATA_ERROR_NOT_INITIALIZED);
    }
    // Gen5's Umount2 places the mount-point pointer in RSI; retain RDI as a compatibility fallback
    // for the older ABI that used the same export name.
    let mount_point_address = if arguments[1] != 0 {
        arguments[1]
    } else {
        arguments[0]
    };
    let Some(mount_point) = read_guest_fixed_ascii(mount_point_address, 16) else {
        return sce_kernel_error(SAVE_DATA_ERROR_PARAMETER);
    };
    let Some(mounted) = state.save_mounts.get(&mount_point) else {
        return sce_kernel_error(SAVE_DATA_ERROR_NOT_FOUND);
    };
    if mounted.resource != 0 && state.prepared_save_resources.contains(&mounted.resource) {
        return sce_kernel_error(SAVE_DATA_ERROR_BUSY);
    }
    state.save_mounts.remove(&mount_point);
    eprintln!("[REMu][SAVEDATA] unmounted {mount_point:?}");
    0
}

fn save_data_prepare(arguments: [u64; 6], state: &mut ProbeDispatchState) -> u64 {
    if !state.save_data_initialized {
        return sce_kernel_error(SAVE_DATA_ERROR_NOT_INITIALIZED);
    }
    let (mount_point_address, prepare_address) = (arguments[0], arguments[1]);
    if mount_point_address == 0 || prepare_address == 0 {
        return sce_kernel_error(SAVE_DATA_ERROR_PARAMETER);
    }
    let Some(mount_point) = read_guest_fixed_ascii(mount_point_address, 16) else {
        return sce_kernel_error(SAVE_DATA_ERROR_PARAMETER);
    };
    let resource = read_guest_u32(prepare_address);
    if resource == 0 || !state.save_transaction_resources.contains(&resource) {
        return sce_kernel_error(SAVE_DATA_ERROR_PARAMETER);
    }
    let Some(mounted) = state.save_mounts.get(&mount_point) else {
        return sce_kernel_error(SAVE_DATA_ERROR_NOT_MOUNTED);
    };
    if mounted.read_only {
        return sce_kernel_error(SAVE_DATA_ERROR_BAD_MOUNTED);
    }
    if mounted.resource != resource {
        return sce_kernel_error(SAVE_DATA_ERROR_PARAMETER);
    }
    if state.prepared_save_resources.contains(&resource) {
        return sce_kernel_error(SAVE_DATA_ERROR_BUSY);
    }

    let committed_path = mounted.committed_path.clone();
    let staging_path = state.save_root.join(format!(
        "sce_transaction_{}_{}",
        std::process::id(),
        resource
    ));
    if staging_path.exists() {
        return sce_kernel_error(SAVE_DATA_ERROR_BUSY);
    }
    if let Err(error) = copy_save_data_tree(&committed_path, &staging_path) {
        eprintln!(
            "[REMu][SAVEDATA] prepare failed resource={resource} staging={:?}: {error}",
            staging_path.display()
        );
        return sce_kernel_error(SAVE_DATA_ERROR_INTERNAL);
    }
    let mounted = state
        .save_mounts
        .get_mut(&mount_point)
        .expect("validated save mount remains live under dispatcher lock");
    mounted.path.clone_from(&staging_path);
    state.prepared_save_resources.insert(resource);
    state.save_transactions.insert(
        resource,
        SaveDataTransaction {
            mount_point: mount_point.clone(),
            committed_path,
            staging_path,
        },
    );
    eprintln!("[REMu][SAVEDATA] prepared mount={mount_point:?} resource={resource}");
    0
}

fn save_data_set_param(arguments: [u64; 6], state: &mut ProbeDispatchState) -> u64 {
    if !state.save_data_initialized {
        return sce_kernel_error(SAVE_DATA_ERROR_NOT_INITIALIZED);
    }
    let (mount_point_address, param_type, param_address, param_size) =
        (arguments[0], arguments[1], arguments[2], arguments[3]);
    if mount_point_address == 0 || param_address == 0 || param_type > 4 {
        return sce_kernel_error(SAVE_DATA_ERROR_PARAMETER);
    }
    let Some(mount_point) = read_guest_fixed_ascii(mount_point_address, 16) else {
        return sce_kernel_error(SAVE_DATA_ERROR_PARAMETER);
    };
    let Some(mounted) = state.save_mounts.get(&mount_point) else {
        return sce_kernel_error(SAVE_DATA_ERROR_NOT_MOUNTED);
    };
    if mounted.read_only {
        return sce_kernel_error(SAVE_DATA_ERROR_BAD_MOUNTED);
    }
    let mut param = match read_save_data_param(&mounted.path, &mounted.dir_name) {
        Ok(param) => param,
        Err(error) => {
            eprintln!("[REMu][SAVEDATA] read param failed: {error}");
            return sce_kernel_error(SAVE_DATA_ERROR_INTERNAL);
        }
    };

    match param_type {
        0 => {
            if param_size != SAVE_DATA_PARAM_SIZE as u64 {
                return sce_kernel_error(SAVE_DATA_ERROR_PARAMETER);
            }
            read_guest_bytes(param_address, &mut param);
        }
        1 => {
            if !replace_save_data_string_field(&mut param[0x00..0x80], param_address, param_size) {
                return sce_kernel_error(SAVE_DATA_ERROR_PARAMETER);
            }
        }
        2 => {
            if !replace_save_data_string_field(&mut param[0x80..0x100], param_address, param_size) {
                return sce_kernel_error(SAVE_DATA_ERROR_PARAMETER);
            }
        }
        3 => {
            if !replace_save_data_string_field(&mut param[0x100..0x500], param_address, param_size)
            {
                return sce_kernel_error(SAVE_DATA_ERROR_PARAMETER);
            }
        }
        4 => {
            if param_size != 4 {
                return sce_kernel_error(SAVE_DATA_ERROR_PARAMETER);
            }
            param[0x500..0x504].copy_from_slice(&read_guest_u32(param_address).to_le_bytes());
        }
        _ => unreachable!("validated save-data parameter type"),
    }
    param[0x508..0x510].copy_from_slice(&current_unix_seconds().to_le_bytes());
    if let Err(error) = write_save_data_param(&mounted.path, &param) {
        eprintln!("[REMu][SAVEDATA] write param failed: {error}");
        return sce_kernel_error(SAVE_DATA_ERROR_INTERNAL);
    }
    eprintln!(
        "[REMu][SAVEDATA] set_param mount={mount_point:?} type={param_type} size={param_size:#x}"
    );
    0
}

fn save_data_save_icon(arguments: [u64; 6], state: &ProbeDispatchState) -> u64 {
    if !state.save_data_initialized {
        return sce_kernel_error(SAVE_DATA_ERROR_NOT_INITIALIZED);
    }
    let (mount_point_address, icon_address) = (arguments[0], arguments[1]);
    if mount_point_address == 0 || icon_address == 0 {
        return sce_kernel_error(SAVE_DATA_ERROR_PARAMETER);
    }
    let Some(mount_point) = read_guest_fixed_ascii(mount_point_address, 16) else {
        return sce_kernel_error(SAVE_DATA_ERROR_PARAMETER);
    };
    let Some(mounted) = state.save_mounts.get(&mount_point) else {
        return sce_kernel_error(SAVE_DATA_ERROR_NOT_MOUNTED);
    };
    if mounted.read_only {
        return sce_kernel_error(SAVE_DATA_ERROR_BAD_MOUNTED);
    }
    let buffer_address = read_guest_u64(icon_address);
    let buffer_size = read_guest_u64(icon_address + 0x08);
    let data_size = read_guest_u64(icon_address + 0x10);
    let length = buffer_size.min(data_size);
    if buffer_address == 0 || length > SAVE_DATA_MAX_ICON_SIZE {
        return sce_kernel_error(SAVE_DATA_ERROR_PARAMETER);
    }
    let Ok(length) = usize::try_from(length) else {
        return sce_kernel_error(SAVE_DATA_ERROR_PARAMETER);
    };
    let mut icon = vec![0_u8; length];
    read_guest_bytes(buffer_address, &mut icon);
    let icon_path = mounted.path.join("sce_sys").join("icon0.png");
    if let Err(error) = write_host_file_durable(&icon_path, &icon) {
        eprintln!("[REMu][SAVEDATA] write icon failed: {error}");
        return sce_kernel_error(SAVE_DATA_ERROR_INTERNAL);
    }
    eprintln!("[REMu][SAVEDATA] save_icon mount={mount_point:?} bytes={length:#x}");
    0
}

fn save_data_commit(arguments: [u64; 6], state: &mut ProbeDispatchState) -> u64 {
    if !state.save_data_initialized {
        return sce_kernel_error(SAVE_DATA_ERROR_NOT_INITIALIZED);
    }
    let commit_address = arguments[0];
    if commit_address == 0 {
        return sce_kernel_error(SAVE_DATA_ERROR_PARAMETER);
    }
    let resource = read_guest_u32(commit_address);
    if !state.save_transaction_resources.contains(&resource) {
        return sce_kernel_error(SAVE_DATA_ERROR_PARAMETER);
    }
    let Some(transaction) = state.save_transactions.get(&resource).cloned() else {
        return sce_kernel_error(SAVE_DATA_ERROR_BUSY);
    };
    if let Err(error) = commit_save_data_transaction(&transaction) {
        eprintln!("[REMu][SAVEDATA] commit failed resource={resource}: {error}");
        return sce_kernel_error(SAVE_DATA_ERROR_INTERNAL);
    }
    let Some(mounted) = state.save_mounts.get_mut(&transaction.mount_point) else {
        return sce_kernel_error(SAVE_DATA_ERROR_NOT_MOUNTED);
    };
    mounted.path.clone_from(&transaction.committed_path);
    state.save_transactions.remove(&resource);
    state.prepared_save_resources.remove(&resource);
    eprintln!("[REMu][SAVEDATA] committed resource={resource}");
    0
}

fn create_save_data_slot(slot: &Path, dir_name: &str, blocks: u64) -> std::io::Result<()> {
    std::fs::create_dir_all(slot)?;
    let param = default_save_data_param(dir_name, current_unix_seconds());
    write_save_data_param(slot, &param)?;
    write_host_file_durable(
        &slot.join("sce_sys").join("blocks.bin"),
        &blocks.to_le_bytes(),
    )
}

fn enumerate_save_data_entries(
    root: &Path,
    pattern: &str,
) -> std::io::Result<Vec<SaveDataSearchEntry>> {
    if !root.is_dir() {
        return Ok(Vec::new());
    }
    let mut entries = Vec::new();
    for item in std::fs::read_dir(root)? {
        let item = item?;
        if !item.file_type()?.is_dir() {
            continue;
        }
        let Some(name) = item.file_name().to_str().map(ToOwned::to_owned) else {
            continue;
        };
        if name.to_ascii_lowercase().starts_with("sce_")
            || (!pattern.is_empty() && !save_data_pattern_matches(&name, pattern))
        {
            continue;
        }
        let path = item.path();
        let (size, modified_seconds) = save_data_directory_stats(&path)?;
        let used_blocks = size.div_ceil(SAVE_DATA_BLOCK_SIZE);
        let stored_blocks = read_save_data_blocks(&path)?.unwrap_or(SAVE_DATA_MIN_BLOCKS);
        let blocks = stored_blocks.max(used_blocks).max(1);
        let mut param = read_save_data_param(&path, &name)?;
        param[0x508..0x510].copy_from_slice(&modified_seconds.to_le_bytes());
        let user_param = u32::from_le_bytes(
            param[0x500..0x504]
                .try_into()
                .expect("fixed save user-param field"),
        );
        entries.push(SaveDataSearchEntry {
            name,
            param,
            blocks,
            free_blocks: blocks.saturating_sub(used_blocks),
            modified_seconds,
            user_param,
        });
    }
    Ok(entries)
}

fn sort_save_data_entries(entries: &mut [SaveDataSearchEntry], sort_key: u32, sort_order: u32) {
    entries.sort_by(|left, right| {
        let order = match sort_key {
            0 => left
                .name
                .to_ascii_lowercase()
                .cmp(&right.name.to_ascii_lowercase()),
            1 => left.user_param.cmp(&right.user_param),
            2 => left.blocks.cmp(&right.blocks),
            3 => left.modified_seconds.cmp(&right.modified_seconds),
            5 => left.free_blocks.cmp(&right.free_blocks),
            _ => unreachable!("validated save-data sort key"),
        };
        order.then_with(|| {
            left.name
                .to_ascii_lowercase()
                .cmp(&right.name.to_ascii_lowercase())
        })
    });
    if sort_order == 1 {
        entries.reverse();
    }
}

fn save_data_pattern_matches(value: &str, pattern: &str) -> bool {
    let value = value
        .chars()
        .map(|character| character.to_ascii_lowercase())
        .collect::<Vec<_>>();
    let pattern = pattern
        .chars()
        .map(|character| character.to_ascii_lowercase())
        .collect::<Vec<_>>();
    let mut previous = vec![false; value.len() + 1];
    previous[0] = true;
    for token in pattern {
        let mut current = vec![false; value.len() + 1];
        if token == '%' {
            current[0] = previous[0];
            for index in 1..=value.len() {
                current[index] = previous[index] || current[index - 1];
            }
        } else {
            for index in 1..=value.len() {
                current[index] = previous[index - 1] && (token == '_' || token == value[index - 1]);
            }
        }
        previous = current;
    }
    previous[value.len()]
}

fn default_save_data_param(dir_name: &str, modified_seconds: i64) -> [u8; SAVE_DATA_PARAM_SIZE] {
    let mut param = [0_u8; SAVE_DATA_PARAM_SIZE];
    write_ascii_slice(&mut param[0x00..0x80], dir_name);
    param[0x508..0x510].copy_from_slice(&modified_seconds.to_le_bytes());
    param
}

fn read_save_data_param(
    slot: &Path,
    dir_name: &str,
) -> std::io::Result<[u8; SAVE_DATA_PARAM_SIZE]> {
    let path = slot.join("sce_sys").join("param.bin");
    if !path.exists() {
        let modified = std::fs::metadata(slot)
            .and_then(|metadata| metadata.modified())
            .map_or_else(|_| current_unix_seconds(), system_time_unix_seconds);
        return Ok(default_save_data_param(dir_name, modified));
    }
    let bytes = std::fs::read(path)?;
    bytes.try_into().map_err(|bytes: Vec<u8>| {
        std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            format!(
                "save parameter has {} bytes, expected {SAVE_DATA_PARAM_SIZE}",
                bytes.len()
            ),
        )
    })
}

fn write_save_data_param(slot: &Path, param: &[u8; SAVE_DATA_PARAM_SIZE]) -> std::io::Result<()> {
    write_host_file_durable(&slot.join("sce_sys").join("param.bin"), param)
}

fn read_save_data_blocks(slot: &Path) -> std::io::Result<Option<u64>> {
    let path = slot.join("sce_sys").join("blocks.bin");
    if !path.exists() {
        return Ok(None);
    }
    let bytes = std::fs::read(path)?;
    let bytes: [u8; 8] = bytes.try_into().map_err(|bytes: Vec<u8>| {
        std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            format!("save block metadata has {} bytes, expected 8", bytes.len()),
        )
    })?;
    Ok(Some(u64::from_le_bytes(bytes)))
}

fn replace_save_data_string_field(field: &mut [u8], address: u64, size: u64) -> bool {
    let Ok(size) = usize::try_from(size) else {
        return false;
    };
    if size == 0 || size > field.len() {
        return false;
    }
    let mut source = vec![0_u8; size];
    read_guest_bytes(address, &mut source);
    field.fill(0);
    let source_length = source.iter().position(|byte| *byte == 0).unwrap_or(size);
    let copy_length = source_length.min(field.len().saturating_sub(1));
    field[..copy_length].copy_from_slice(&source[..copy_length]);
    true
}

fn write_ascii_slice(destination: &mut [u8], value: &str) {
    destination.fill(0);
    let copy_length = value.len().min(destination.len().saturating_sub(1));
    for (output, input) in destination[..copy_length].iter_mut().zip(value.as_bytes()) {
        *output = if input.is_ascii() { *input } else { b'?' };
    }
}

fn read_guest_fixed_ascii(address: u64, length: usize) -> Option<String> {
    if address == 0 {
        return None;
    }
    // SAFETY: fixed-width SaveData ABI strings point to readable guest records of `length` bytes.
    let bytes = unsafe { std::slice::from_raw_parts(address as *const u8, length) };
    let end = bytes.iter().position(|byte| *byte == 0).unwrap_or(length);
    if !bytes[..end].iter().all(u8::is_ascii) {
        return None;
    }
    String::from_utf8(bytes[..end].to_vec()).ok()
}

fn write_guest_fixed_ascii(address: u64, length: usize, value: &str) {
    // SAFETY: the SaveData ABI supplies a writable fixed-width result record.
    let destination = unsafe { std::slice::from_raw_parts_mut(address as *mut u8, length) };
    write_ascii_slice(destination, value);
}

fn read_guest_bytes(address: u64, destination: &mut [u8]) {
    // SAFETY: the guest ABI supplies a readable range matching the declared buffer size.
    unsafe {
        std::ptr::copy_nonoverlapping(
            address as *const u8,
            destination.as_mut_ptr(),
            destination.len(),
        );
    }
}

fn write_guest_bytes(address: u64, source: &[u8]) {
    // SAFETY: the guest ABI supplies a writable range matching the declared result capacity.
    unsafe {
        std::ptr::copy_nonoverlapping(source.as_ptr(), address as *mut u8, source.len());
    }
}

fn save_data_directory_stats(root: &Path) -> std::io::Result<(u64, i64)> {
    let root_metadata = std::fs::metadata(root)?;
    let mut total = 0_u64;
    let mut latest = root_metadata
        .modified()
        .map_or_else(|_| current_unix_seconds(), system_time_unix_seconds);
    let mut pending = vec![root.to_owned()];
    while let Some(directory) = pending.pop() {
        for entry in std::fs::read_dir(directory)? {
            let entry = entry?;
            let metadata = entry.metadata()?;
            latest = latest.max(
                metadata
                    .modified()
                    .map_or_else(|_| latest, system_time_unix_seconds),
            );
            if metadata.is_dir() {
                pending.push(entry.path());
            } else if metadata.is_file() {
                total = total.saturating_add(metadata.len());
            }
        }
    }
    Ok((total, latest))
}

fn current_unix_seconds() -> i64 {
    system_time_unix_seconds(std::time::SystemTime::now())
}

fn system_time_unix_seconds(time: std::time::SystemTime) -> i64 {
    time.duration_since(std::time::UNIX_EPOCH).map_or_else(
        |error| -i64::try_from(error.duration().as_secs()).unwrap_or(i64::MAX),
        |duration| i64::try_from(duration.as_secs()).unwrap_or(i64::MAX),
    )
}

fn write_host_file_durable(path: &Path, bytes: &[u8]) -> std::io::Result<()> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let mut file = std::fs::OpenOptions::new()
        .create(true)
        .truncate(true)
        .write(true)
        .open(path)?;
    file.write_all(bytes)?;
    file.sync_all()
}

fn copy_save_data_tree(source: &Path, destination: &Path) -> std::io::Result<()> {
    std::fs::create_dir(destination)?;
    let mut pending = vec![(source.to_owned(), destination.to_owned())];
    while let Some((source_directory, destination_directory)) = pending.pop() {
        for entry in std::fs::read_dir(source_directory)? {
            let entry = entry?;
            let source_path = entry.path();
            let destination_path = destination_directory.join(entry.file_name());
            let file_type = entry.file_type()?;
            if file_type.is_dir() {
                std::fs::create_dir(&destination_path)?;
                pending.push((source_path, destination_path));
            } else if file_type.is_file() {
                std::fs::copy(source_path, destination_path)?;
            }
        }
    }
    Ok(())
}

fn sync_save_data_tree(root: &Path) -> std::io::Result<()> {
    let mut directories = vec![root.to_owned()];
    let mut index = 0;
    while index < directories.len() {
        for entry in std::fs::read_dir(&directories[index])? {
            let entry = entry?;
            let file_type = entry.file_type()?;
            if file_type.is_dir() {
                directories.push(entry.path());
            } else if file_type.is_file() {
                std::fs::File::open(entry.path())?.sync_all()?;
            }
        }
        index += 1;
    }
    for directory in directories.into_iter().rev() {
        std::fs::File::open(directory)?.sync_all()?;
    }
    Ok(())
}

fn commit_save_data_transaction(transaction: &SaveDataTransaction) -> std::io::Result<()> {
    sync_save_data_tree(&transaction.staging_path)?;
    let file_name = transaction
        .committed_path
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or("save");
    let backup_path = transaction.committed_path.with_file_name(format!(
        "sce_commit_backup_{}_{}_{}",
        std::process::id(),
        file_name,
        transaction
            .staging_path
            .file_name()
            .and_then(|name| name.to_str())
            .unwrap_or("transaction")
    ));
    std::fs::rename(&transaction.committed_path, &backup_path)?;
    if let Err(error) = std::fs::rename(&transaction.staging_path, &transaction.committed_path) {
        let _ = std::fs::rename(&backup_path, &transaction.committed_path);
        return Err(error);
    }
    if let Some(parent) = transaction.committed_path.parent() {
        std::fs::File::open(parent)?.sync_all()?;
    }
    std::fs::remove_dir_all(backup_path)
}

fn guest_sincosf(value: f32, sin_address: u64, cos_address: u64) -> u64 {
    let (sin, cos) = value.sin_cos();
    if sin_address != 0 {
        write_guest_u32(sin_address, sin.to_bits());
    }
    if cos_address != 0 {
        write_guest_u32(cos_address, cos.to_bits());
    }
    0
}

fn guest_qsort(arguments: [u64; 6]) -> u64 {
    let (base, count, size, comparator) = (arguments[0], arguments[1], arguments[2], arguments[3]);
    let (Ok(count), Ok(size), Ok(comparator)) = (
        usize::try_from(count),
        usize::try_from(size),
        usize::try_from(comparator),
    ) else {
        return 0;
    };
    if base == 0
        || comparator == 0
        || count < 2
        || size == 0
        || count > 1_000_000
        || size > 1_048_576
    {
        return 0;
    }
    let Ok(base) = usize::try_from(base) else {
        return 0;
    };
    // The guest comparator is native x86-64 code in this Rosetta process. Dropping the dispatcher
    // lock before entering this function permits a comparator that itself crosses an import stub.
    let compare: unsafe extern "C" fn(*const u8, *const u8) -> i32 =
        unsafe { std::mem::transmute(comparator) };
    let mut key = vec![0_u8; size];
    for index in 1..count {
        // SAFETY: qsort's ABI guarantees a writable `count * size` byte range at `base`.
        unsafe {
            let current = (base + index * size) as *mut u8;
            std::ptr::copy_nonoverlapping(current, key.as_mut_ptr(), size);
            let mut insertion = index;
            while insertion > 0 {
                let previous = (base + (insertion - 1) * size) as *mut u8;
                if compare(previous, key.as_ptr()) <= 0 {
                    break;
                }
                std::ptr::copy(previous, previous.add(size), size);
                insertion -= 1;
            }
            std::ptr::copy_nonoverlapping(key.as_ptr(), (base + insertion * size) as *mut u8, size);
        }
    }
    0
}

fn guest_fopen(path_address: u64, mode_address: u64, state: &ProbeDispatchState) -> u64 {
    let (Some(guest_path), Some(mode)) = (
        read_guest_cstring(path_address),
        read_guest_cstring(mode_address),
    ) else {
        return 0;
    };
    let Some((host_path, writable)) = resolve_probe_guest_path(state, &guest_path) else {
        eprintln!("[REMu][VFS] rejected path {guest_path:?}");
        return 0;
    };
    if !mode.starts_with('r') && !writable {
        eprintln!("[REMu][VFS] rejected write mode {mode:?} for {guest_path:?}");
        return 0;
    }
    if writable
        && !mode.starts_with('r')
        && let Some(parent) = host_path.parent()
        && std::fs::create_dir_all(parent).is_err()
    {
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
    resolve_path_beneath(game_root, relative)
}

fn resolve_probe_guest_path(
    state: &ProbeDispatchState,
    guest_path: &str,
) -> Option<(PathBuf, bool)> {
    let normalized = guest_path.replace('\\', "/");
    for (mount_point, mounted) in &state.save_mounts {
        if normalized == *mount_point {
            return Some((mounted.path.clone(), !mounted.read_only));
        }
        if let Some(relative) = normalized.strip_prefix(&format!("{mount_point}/")) {
            return resolve_path_beneath(&mounted.path, relative)
                .map(|path| (path, !mounted.read_only));
        }
    }
    resolve_guest_path(&state.game_root, &normalized).map(|path| (path, false))
}

fn resolve_path_beneath(root: &Path, relative: &str) -> Option<PathBuf> {
    let mut host = root.to_owned();
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

const SCE_KERNEL_ERROR_PERMISSION_DENIED: u32 = 0x8002_000d;
const SCE_KERNEL_ERROR_NOT_FOUND: u32 = 0x8002_0002;
const SCE_KERNEL_ERROR_INTERRUPTED: u32 = 0x8002_0004;
const SCE_KERNEL_ERROR_IO: u32 = 0x8002_0005;
const SCE_KERNEL_ERROR_BAD_FILE_DESCRIPTOR: u32 = 0x8002_0009;
const SCE_KERNEL_ERROR_OUT_OF_MEMORY: u32 = 0x8002_000c;
const SCE_KERNEL_ERROR_MEMORY_FAULT: u32 = 0x8002_000e;
const SCE_KERNEL_ERROR_BUSY: u32 = 0x8002_0010;
const SCE_KERNEL_ERROR_ALREADY_EXISTS: u32 = 0x8002_0011;
const SCE_KERNEL_ERROR_CROSS_DEVICE: u32 = 0x8002_0012;
const SCE_KERNEL_ERROR_NOT_DIRECTORY: u32 = 0x8002_0014;
const SCE_KERNEL_ERROR_IS_DIRECTORY: u32 = 0x8002_0015;
const SCE_KERNEL_ERROR_INVALID_ARGUMENT: u32 = 0x8002_0016;
const SCE_KERNEL_ERROR_FILE_TOO_LARGE: u32 = 0x8002_001b;
const SCE_KERNEL_ERROR_NO_SPACE: u32 = 0x8002_001c;
const SCE_KERNEL_ERROR_READ_ONLY: u32 = 0x8002_001e;
const SCE_KERNEL_ERROR_BROKEN_PIPE: u32 = 0x8002_0020;
const SCE_KERNEL_ERROR_TRY_AGAIN: u32 = 0x8002_0023;
const SCE_KERNEL_ERROR_NOT_SUPPORTED: u32 = 0x8002_002d;
const SCE_KERNEL_ERROR_NOT_IMPLEMENTED: u32 = 0x8002_004e;
const SCE_KERNEL_STAT_SIZE: usize = 120;

fn sce_kernel_error(code: u32) -> u64 {
    signed_i32_return(i32::from_ne_bytes(code.to_ne_bytes()))
}

fn guest_posix_file_result(result: u64) -> u64 {
    let code = low_u32(result);
    if code & 0x8000_0000 == 0 {
        return result;
    }
    let errno = if code & 0xffff_ff00 == 0x8002_0000 {
        i32::try_from(code & 0xff).expect("kernel errno fits i32")
    } else {
        libc::EIO
    };
    let errno_address = guest_errno_address();
    if errno_address != 0 {
        write_guest_u32(errno_address, errno.cast_unsigned());
    }
    u64::MAX
}

fn sce_kernel_io_error(error: &std::io::Error) -> u64 {
    use std::io::ErrorKind;

    let code = match error.kind() {
        ErrorKind::NotFound => SCE_KERNEL_ERROR_NOT_FOUND,
        ErrorKind::PermissionDenied => SCE_KERNEL_ERROR_PERMISSION_DENIED,
        ErrorKind::BrokenPipe => SCE_KERNEL_ERROR_BROKEN_PIPE,
        ErrorKind::AlreadyExists => SCE_KERNEL_ERROR_ALREADY_EXISTS,
        ErrorKind::WouldBlock => SCE_KERNEL_ERROR_TRY_AGAIN,
        ErrorKind::InvalidInput | ErrorKind::InvalidData => SCE_KERNEL_ERROR_INVALID_ARGUMENT,
        ErrorKind::TimedOut => 0x8002_003c,
        ErrorKind::Interrupted => SCE_KERNEL_ERROR_INTERRUPTED,
        ErrorKind::Unsupported => SCE_KERNEL_ERROR_NOT_SUPPORTED,
        ErrorKind::OutOfMemory => SCE_KERNEL_ERROR_OUT_OF_MEMORY,
        ErrorKind::NotADirectory => SCE_KERNEL_ERROR_NOT_DIRECTORY,
        ErrorKind::IsADirectory => SCE_KERNEL_ERROR_IS_DIRECTORY,
        ErrorKind::DirectoryNotEmpty => 0x8002_0042,
        ErrorKind::ReadOnlyFilesystem => SCE_KERNEL_ERROR_READ_ONLY,
        ErrorKind::StaleNetworkFileHandle => 0x8002_0046,
        ErrorKind::StorageFull | ErrorKind::QuotaExceeded => SCE_KERNEL_ERROR_NO_SPACE,
        ErrorKind::FileTooLarge => SCE_KERNEL_ERROR_FILE_TOO_LARGE,
        ErrorKind::ResourceBusy | ErrorKind::ExecutableFileBusy => SCE_KERNEL_ERROR_BUSY,
        ErrorKind::Deadlock => 0x8002_000b,
        ErrorKind::CrossesDevices => SCE_KERNEL_ERROR_CROSS_DEVICE,
        ErrorKind::TooManyLinks => 0x8002_001f,
        ErrorKind::InvalidFilename => 0x8002_003f,
        ErrorKind::ArgumentListTooLong => 0x8002_0007,
        _ => SCE_KERNEL_ERROR_IO,
    };
    sce_kernel_error(code)
}

fn audio_out_error(error: audio::AudioOutError) -> u64 {
    let code = match error {
        audio::AudioOutError::NotOpened => 0x8026_0001,
        audio::AudioOutError::Busy => 0x8026_0002,
        audio::AudioOutError::InvalidPort => 0x8026_0003,
        audio::AudioOutError::InvalidPointer => 0x8026_0004,
        audio::AudioOutError::PortFull => 0x8026_0005,
        audio::AudioOutError::InvalidSize => 0x8026_0006,
        audio::AudioOutError::InvalidFormat => 0x8026_0007,
        audio::AudioOutError::InvalidSampleFrequency => 0x8026_0008,
        audio::AudioOutError::InvalidVolume => 0x8026_0009,
        audio::AudioOutError::InvalidPortType => 0x8026_000a,
        audio::AudioOutError::OutOfMemory => 0x8026_000d,
    };
    sce_kernel_error(code)
}

fn guest_kernel_open(arguments: [u64; 6], state: &mut ProbeDispatchState) -> u64 {
    let Some(guest_path) = read_guest_cstring(arguments[0]) else {
        return sce_kernel_error(SCE_KERNEL_ERROR_MEMORY_FAULT);
    };
    let flags = low_i32(arguments[1]);
    let access = flags & 0x3;
    if access == 0x3 {
        return sce_kernel_error(SCE_KERNEL_ERROR_INVALID_ARGUMENT);
    }
    let Some((host_path, writable)) = resolve_probe_guest_path(state, &guest_path) else {
        eprintln!("[REMu][VFS] sceKernelOpen rejected guest={guest_path:?}");
        return sce_kernel_error(SCE_KERNEL_ERROR_NOT_FOUND);
    };
    if access != 0 && !writable {
        eprintln!(
            "[REMu][VFS] sceKernelOpen rejected write access guest={guest_path:?} flags={flags:#x}"
        );
        return sce_kernel_error(SCE_KERNEL_ERROR_PERMISSION_DENIED);
    }
    let create = writable && flags & 0x200 != 0;
    let exclusive = create && flags & 0x800 != 0;
    let mut options = std::fs::OpenOptions::new();
    options
        .read(access != 1)
        .write(access != 0)
        .append(access != 0 && flags & 0x8 != 0)
        .create(create && !exclusive)
        .create_new(exclusive)
        .truncate(writable && access != 0 && flags & 0x400 != 0);
    #[cfg(unix)]
    {
        options.mode(low_u32(arguments[2]));
    }
    let file = match options.open(&host_path) {
        Ok(file) => file,
        Err(error) => {
            eprintln!(
                "[REMu][VFS] sceKernelOpen failed guest={guest_path:?} host={:?}: {error}",
                host_path.display()
            );
            return sce_kernel_io_error(&error);
        }
    };
    if flags & 0x0002_0000 != 0 && file.metadata().is_ok_and(|metadata| !metadata.is_dir()) {
        eprintln!(
            "[REMu][VFS] sceKernelOpen expected directory guest={guest_path:?} host={:?}",
            host_path.display()
        );
        return sce_kernel_error(SCE_KERNEL_ERROR_NOT_DIRECTORY);
    }
    let fd = state.next_kernel_fd.max(3);
    state.next_kernel_fd = fd.saturating_add(1);
    state.kernel_files.insert(fd, file);
    eprintln!(
        "[REMu][VFS] sceKernelOpen guest={guest_path:?} host={:?} flags={flags:#x} fd={fd}",
        host_path.display()
    );
    signed_i32_return(fd)
}

fn guest_kernel_stat(arguments: [u64; 6], state: &ProbeDispatchState) -> u64 {
    if arguments[1] == 0 {
        return sce_kernel_error(SCE_KERNEL_ERROR_MEMORY_FAULT);
    }
    let Some(guest_path) = read_guest_cstring(arguments[0]) else {
        return sce_kernel_error(SCE_KERNEL_ERROR_MEMORY_FAULT);
    };
    let Some((host_path, _)) = resolve_probe_guest_path(state, &guest_path) else {
        eprintln!("[REMu][VFS] sceKernelStat rejected guest={guest_path:?}");
        return sce_kernel_error(SCE_KERNEL_ERROR_NOT_FOUND);
    };
    let metadata = match std::fs::metadata(&host_path) {
        Ok(metadata) => metadata,
        Err(error) => {
            eprintln!(
                "[REMu][VFS] sceKernelStat failed guest={guest_path:?} host={:?}: {error}",
                host_path.display()
            );
            return sce_kernel_io_error(&error);
        }
    };
    let payload = kernel_stat_payload_from_metadata(&metadata, guest_path.as_bytes());
    // SAFETY: sceKernelStat supplies a writable 120-byte Gen5 stat object in live guest memory.
    unsafe {
        std::ptr::copy_nonoverlapping(payload.as_ptr(), arguments[1] as *mut u8, payload.len());
    }
    eprintln!(
        "[REMu][VFS] sceKernelStat guest={guest_path:?} host={:?} size={}",
        host_path.display(),
        metadata.len()
    );
    0
}

fn guest_kernel_read(arguments: [u64; 6], state: &mut ProbeDispatchState) -> u64 {
    let fd = low_i32(arguments[0]);
    let destination = arguments[1];
    let requested = arguments[2];
    if requested == 0 {
        return 0;
    }
    if destination == 0 {
        return sce_kernel_error(SCE_KERNEL_ERROR_MEMORY_FAULT);
    }
    let Ok(requested) = usize::try_from(requested) else {
        return sce_kernel_error(SCE_KERNEL_ERROR_INVALID_ARGUMENT);
    };
    let Some(file) = state.kernel_files.get_mut(&fd) else {
        return sce_kernel_error(SCE_KERNEL_ERROR_BAD_FILE_DESCRIPTOR);
    };
    // SAFETY: sceKernelRead supplies a writable range of exactly `requested` bytes in live guest
    // memory, and the global import dispatcher serializes access to the associated File.
    let destination = unsafe { std::slice::from_raw_parts_mut(destination as *mut u8, requested) };
    let read = match file.read(destination) {
        Ok(read) => read,
        Err(error) => return sce_kernel_io_error(&error),
    };
    eprintln!("[REMu][VFS] sceKernelRead fd={fd} requested={requested} read={read}");
    u64::try_from(read).expect("read byte count fits u64")
}

fn guest_kernel_write(arguments: [u64; 6], state: &mut ProbeDispatchState) -> u64 {
    let fd = low_i32(arguments[0]);
    let Ok(length) = usize::try_from(arguments[2]) else {
        return sce_kernel_error(SCE_KERNEL_ERROR_INVALID_ARGUMENT);
    };
    if length != 0 && arguments[1] == 0 {
        return sce_kernel_error(SCE_KERNEL_ERROR_MEMORY_FAULT);
    }
    let payload = if length == 0 {
        &[]
    } else {
        // SAFETY: `_write` supplies a readable guest buffer of exactly `length` bytes.
        unsafe { std::slice::from_raw_parts(arguments[1] as *const u8, length) }
    };
    let result = match fd {
        1 => std::io::stdout().write(payload),
        2 => std::io::stderr().write(payload),
        _ => state
            .kernel_files
            .get_mut(&fd)
            .ok_or_else(|| std::io::Error::from(std::io::ErrorKind::NotFound))
            .and_then(|file| file.write(payload)),
    };
    match result {
        Ok(written) => u64::try_from(written).expect("written byte count fits u64"),
        Err(error) if fd != 1 && fd != 2 && !state.kernel_files.contains_key(&fd) => {
            sce_kernel_error(SCE_KERNEL_ERROR_BAD_FILE_DESCRIPTOR)
        }
        Err(error) => sce_kernel_io_error(&error),
    }
}

fn guest_kernel_close(arguments: [u64; 6], state: &mut ProbeDispatchState) -> u64 {
    let fd = low_i32(arguments[0]);
    if state.kernel_files.remove(&fd).is_none() {
        return sce_kernel_error(SCE_KERNEL_ERROR_BAD_FILE_DESCRIPTOR);
    }
    eprintln!("[REMu][VFS] sceKernelClose fd={fd}");
    0
}

fn guest_kernel_lseek(arguments: [u64; 6], state: &mut ProbeDispatchState) -> u64 {
    let fd = low_i32(arguments[0]);
    let offset = i64::from_ne_bytes(arguments[1].to_ne_bytes());
    let seek_from = match low_i32(arguments[2]) {
        0 if offset >= 0 => std::io::SeekFrom::Start(offset.cast_unsigned()),
        1 => std::io::SeekFrom::Current(offset),
        2 => std::io::SeekFrom::End(offset),
        _ => return sce_kernel_error(SCE_KERNEL_ERROR_INVALID_ARGUMENT),
    };
    let Some(file) = state.kernel_files.get_mut(&fd) else {
        return sce_kernel_error(SCE_KERNEL_ERROR_BAD_FILE_DESCRIPTOR);
    };
    match file.seek(seek_from) {
        Ok(position) => position,
        Err(error) => sce_kernel_io_error(&error),
    }
}

fn guest_kernel_fstat(arguments: [u64; 6], state: &ProbeDispatchState) -> u64 {
    if arguments[1] == 0 {
        return sce_kernel_error(SCE_KERNEL_ERROR_MEMORY_FAULT);
    }
    let fd = low_i32(arguments[0]);
    let Some(file) = state.kernel_files.get(&fd) else {
        return sce_kernel_error(SCE_KERNEL_ERROR_BAD_FILE_DESCRIPTOR);
    };
    let metadata = match file.metadata() {
        Ok(metadata) => metadata,
        Err(error) => return sce_kernel_io_error(&error),
    };
    let payload = kernel_stat_payload_from_metadata(&metadata, &fd.to_le_bytes());
    // SAFETY: the guest ABI supplies writable storage for one complete 120-byte stat object.
    unsafe {
        std::ptr::copy_nonoverlapping(payload.as_ptr(), arguments[1] as *mut u8, payload.len());
    }
    0
}

fn guest_kernel_ftruncate(arguments: [u64; 6], state: &mut ProbeDispatchState) -> u64 {
    let fd = low_i32(arguments[0]);
    let length = i64::from_ne_bytes(arguments[1].to_ne_bytes());
    if length < 0 {
        return sce_kernel_error(SCE_KERNEL_ERROR_INVALID_ARGUMENT);
    }
    let Some(file) = state.kernel_files.get_mut(&fd) else {
        return sce_kernel_error(SCE_KERNEL_ERROR_BAD_FILE_DESCRIPTOR);
    };
    file.set_len(length.cast_unsigned())
        .map_or_else(|error| sce_kernel_io_error(&error), |()| 0)
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
struct GuestKernelStat {
    dev: u32,
    ino: u32,
    mode: u16,
    nlink: u16,
    uid: u32,
    gid: u32,
    rdev: u32,
    atime_sec: i64,
    atime_nsec: i64,
    mtime_sec: i64,
    mtime_nsec: i64,
    ctime_sec: i64,
    ctime_nsec: i64,
    size: i64,
    blocks: i64,
    block_size: u32,
    flags: u32,
    generation: u32,
    spare: i32,
    birthtime_sec: i64,
    birthtime_nsec: i64,
}

impl GuestKernelStat {
    fn encode(self) -> [u8; SCE_KERNEL_STAT_SIZE] {
        let mut payload = [0_u8; SCE_KERNEL_STAT_SIZE];
        payload[0..4].copy_from_slice(&self.dev.to_le_bytes());
        payload[4..8].copy_from_slice(&self.ino.to_le_bytes());
        payload[8..10].copy_from_slice(&self.mode.to_le_bytes());
        payload[10..12].copy_from_slice(&self.nlink.to_le_bytes());
        payload[12..16].copy_from_slice(&self.uid.to_le_bytes());
        payload[16..20].copy_from_slice(&self.gid.to_le_bytes());
        payload[20..24].copy_from_slice(&self.rdev.to_le_bytes());
        payload[24..32].copy_from_slice(&self.atime_sec.to_le_bytes());
        payload[32..40].copy_from_slice(&self.atime_nsec.to_le_bytes());
        payload[40..48].copy_from_slice(&self.mtime_sec.to_le_bytes());
        payload[48..56].copy_from_slice(&self.mtime_nsec.to_le_bytes());
        payload[56..64].copy_from_slice(&self.ctime_sec.to_le_bytes());
        payload[64..72].copy_from_slice(&self.ctime_nsec.to_le_bytes());
        payload[72..80].copy_from_slice(&self.size.to_le_bytes());
        payload[80..88].copy_from_slice(&self.blocks.to_le_bytes());
        payload[88..92].copy_from_slice(&self.block_size.to_le_bytes());
        payload[92..96].copy_from_slice(&self.flags.to_le_bytes());
        payload[96..100].copy_from_slice(&self.generation.to_le_bytes());
        payload[100..104].copy_from_slice(&self.spare.to_le_bytes());
        payload[104..112].copy_from_slice(&self.birthtime_sec.to_le_bytes());
        payload[112..120].copy_from_slice(&self.birthtime_nsec.to_le_bytes());
        payload
    }
}

fn fallback_inode(inode_seed: &[u8]) -> u32 {
    inode_seed.iter().fold(0x811c_9dc5_u32, |hash, &byte| {
        (hash ^ u32::from(byte)).wrapping_mul(0x0100_0193)
    })
}

#[cfg(any(test, not(unix)))]
fn kernel_stat_payload(size: u64, is_directory: bool, inode_seed: &[u8]) -> [u8; 120] {
    let size = i64::try_from(size).unwrap_or(i64::MAX);
    GuestKernelStat {
        ino: fallback_inode(inode_seed),
        mode: if is_directory { 0x41ff } else { 0x81ff },
        nlink: 1,
        size,
        blocks: if is_directory {
            128
        } else {
            (size + 511) / 512
        },
        block_size: if is_directory { 65_536 } else { 512 },
        ..GuestKernelStat::default()
    }
    .encode()
}

fn kernel_stat_payload_from_metadata(
    metadata: &std::fs::Metadata,
    inode_seed: &[u8],
) -> [u8; SCE_KERNEL_STAT_SIZE] {
    #[cfg(unix)]
    {
        let (birth_seconds, birth_nanoseconds) = metadata
            .created()
            .ok()
            .and_then(|created| created.duration_since(std::time::UNIX_EPOCH).ok())
            .map_or((metadata.ctime(), metadata.ctime_nsec()), |duration| {
                (
                    i64::try_from(duration.as_secs()).unwrap_or(i64::MAX),
                    i64::from(duration.subsec_nanos()),
                )
            });
        GuestKernelStat {
            dev: u32::try_from(metadata.dev()).unwrap_or(u32::MAX),
            ino: u32::try_from(metadata.ino()).unwrap_or_else(|_| fallback_inode(inode_seed)),
            mode: u16::try_from(metadata.mode() & u32::from(u16::MAX)).expect("masked mode fits"),
            nlink: u16::try_from(metadata.nlink()).unwrap_or(u16::MAX),
            uid: metadata.uid(),
            gid: metadata.gid(),
            rdev: u32::try_from(metadata.rdev()).unwrap_or(u32::MAX),
            atime_sec: metadata.atime(),
            atime_nsec: metadata.atime_nsec(),
            mtime_sec: metadata.mtime(),
            mtime_nsec: metadata.mtime_nsec(),
            ctime_sec: metadata.ctime(),
            ctime_nsec: metadata.ctime_nsec(),
            size: i64::try_from(metadata.size()).unwrap_or(i64::MAX),
            blocks: i64::try_from(metadata.blocks()).unwrap_or(i64::MAX),
            block_size: u32::try_from(metadata.blksize()).unwrap_or(u32::MAX),
            flags: 0,
            generation: 0,
            spare: 0,
            birthtime_sec: birth_seconds,
            birthtime_nsec: birth_nanoseconds,
        }
        .encode()
    }
    #[cfg(not(unix))]
    {
        kernel_stat_payload(metadata.len(), metadata.is_dir(), inode_seed)
    }
}

fn guest_memcmp(left: u64, right: u64, length: u64) -> u64 {
    if length == 0 {
        return 0;
    }
    if left == 0 || right == 0 {
        return u64::MAX;
    }
    let Ok(length) = usize::try_from(length) else {
        return u64::MAX;
    };
    // SAFETY: memcmp's guest ABI contract supplies readable ranges through `length`.
    signed_i32_return(unsafe {
        libc::memcmp(
            left as *const libc::c_void,
            right as *const libc::c_void,
            length,
        )
    })
}

fn guest_memchr(address: u64, character: u64, length: u64) -> u64 {
    if address == 0 || length == 0 {
        return 0;
    }
    let Ok(length) = usize::try_from(length) else {
        return 0;
    };
    // SAFETY: memchr's guest ABI contract supplies a readable range through `length`.
    unsafe {
        libc::memchr(address as *const libc::c_void, low_i32(character), length) as usize as u64
    }
}

const GUEST_WCHAR_SIZE: usize = std::mem::size_of::<u16>();

fn guest_wide_byte_length(count: u64) -> Option<usize> {
    usize::try_from(count).ok()?.checked_mul(GUEST_WCHAR_SIZE)
}

fn guest_wmemcpy(destination: u64, source: u64, count: u64) -> u64 {
    guest_wide_copy(destination, source, count)
}

fn guest_wmemmove(destination: u64, source: u64, count: u64) -> u64 {
    guest_wide_copy(destination, source, count)
}

fn guest_wide_copy(destination: u64, source: u64, count: u64) -> u64 {
    let Some(length) = guest_wide_byte_length(count) else {
        return 0;
    };
    if length == 0 {
        return destination;
    }
    if destination == 0 || source == 0 {
        return 0;
    }
    // SAFETY: the guest ABI supplies readable and writable ranges for `count` two-byte wide
    // characters. Gen5's libc copy is overlap-tolerant in the basic_string path used by this
    // title, so both wide-copy entry points intentionally use `copy`.
    unsafe {
        std::ptr::copy(source as *const u8, destination as *mut u8, length);
    }
    destination
}

fn guest_wmemchr(address: u64, character: u64, count: u64) -> u64 {
    if address == 0 || count == 0 {
        return 0;
    }
    let Ok(count) = usize::try_from(count) else {
        return 0;
    };
    let needle =
        u16::try_from(character & u64::from(u16::MAX)).expect("masked wide character fits u16");
    for index in 0..count {
        let offset = index * GUEST_WCHAR_SIZE;
        // SAFETY: wmemchr's guest ABI supplies `count` readable two-byte elements. Unaligned
        // reads keep the HLE correct even when a guest passes packed storage.
        let value: u16 =
            unsafe { std::ptr::read_unaligned((address as *const u8).add(offset).cast()) };
        if value == needle {
            return address + u64::try_from(offset).unwrap_or(u64::MAX);
        }
    }
    0
}

fn guest_wmemcmp(left: u64, right: u64, count: u64) -> u64 {
    if count == 0 {
        return 0;
    }
    if left == 0 || right == 0 {
        return u64::MAX;
    }
    let Ok(count) = usize::try_from(count) else {
        return u64::MAX;
    };
    for index in 0..count {
        let offset = index * GUEST_WCHAR_SIZE;
        // SAFETY: wmemcmp's guest ABI supplies `count` readable two-byte elements on each side.
        let left_value =
            unsafe { std::ptr::read_unaligned((left as *const u8).add(offset).cast::<u16>()) };
        // SAFETY: same contract as the left-side read above.
        let right_value =
            unsafe { std::ptr::read_unaligned((right as *const u8).add(offset).cast::<u16>()) };
        match left_value.cmp(&right_value) {
            std::cmp::Ordering::Less => return signed_i32_return(-1),
            std::cmp::Ordering::Greater => return signed_i32_return(1),
            std::cmp::Ordering::Equal => {}
        }
    }
    0
}

fn guest_sprintf(arguments: [u64; 6]) -> u64 {
    if arguments[0] == 0 || arguments[1] == 0 {
        return u64::MAX;
    }
    // SAFETY: The guest supplies a writable destination, a valid format string, and SysV integer
    // or pointer varargs. Passing all four available argument registers preserves their bit layout.
    signed_i32_return(unsafe {
        libc::sprintf(
            arguments[0] as *mut libc::c_char,
            arguments[1] as *const libc::c_char,
            arguments[2],
            arguments[3],
            arguments[4],
            arguments[5],
        )
    })
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

fn guest_clock_gettime(clock_id: u64, output: u64) -> u64 {
    if output == 0 {
        return u64::MAX;
    }
    // Console clock ids do not share Darwin's numeric constants. This POSIX compatibility export
    // returns wall-clock time, matching the console runtime's observable calendar-time contract.
    let mut timestamp = libc::timespec {
        tv_sec: 0,
        tv_nsec: 0,
    };
    // SAFETY: timestamp is a valid writable host timespec.
    let status = unsafe { libc::clock_gettime(libc::CLOCK_REALTIME, &raw mut timestamp) };
    if status == 0 {
        write_guest_u64(output, u64::try_from(timestamp.tv_sec).unwrap_or(0));
        write_guest_u64(output + 8, u64::try_from(timestamp.tv_nsec).unwrap_or(0));
    }
    if trace_guest_time() {
        eprintln!(
            "[REMu][TIME] clock_gettime id={} output={output:#x} status={status} sec={} nsec={}",
            clock_id, timestamp.tv_sec, timestamp.tv_nsec
        );
    }
    signed_i32_return(status)
}

fn guest_time(output: u64) -> u64 {
    let seconds = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |elapsed| elapsed.as_secs());
    if output != 0 {
        write_guest_u64(output, seconds);
    }
    if trace_guest_time() {
        eprintln!("[REMu][TIME] time output={output:#x} sec={seconds}");
    }
    seconds
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
type GuestModuleInitializer = unsafe extern "C" fn(u64, u64, u64);

#[cfg(target_arch = "x86_64")]
fn probe_entry(
    image: &LinkedImage,
    process_image_name: &str,
    game_root: &Path,
    maximum_import_hits: usize,
    timeout_seconds: u32,
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
    let registered_fdes = register_guest_unwind_frames(image)?;
    if registered_fdes != 0 {
        eprintln!("[REMu][CXX] registered {registered_fdes} guest unwind records");
    }
    configure_probe_dispatch(image, game_root, maximum_import_hits);
    set_current_guest_thread_pointer(image.tls.thread_pointer.get());
    for initializer in &image.module_initializers {
        eprintln!("[REMu][MODULE] initializing runtime at {initializer}");
        // SAFETY: The module linker relocated this DT_INIT address in executable guest memory.
        // Gen5 runtime initializers use the SysV three-argument module-entry convention.
        let initializer: GuestModuleInitializer = unsafe { std::mem::transmute(initializer.get()) };
        // SAFETY: Runtime initializers accept null process-vector arguments and initialize their
        // own module-local data before the main executable starts.
        unsafe {
            initializer(0, 0, 0);
        }
    }
    if timeout_seconds != 0 {
        // SAFETY: Alarm only bounds this diagnostic process; its default action terminates it.
        unsafe {
            libc::alarm(timeout_seconds);
        }
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
    let import_hits = IMPORT_TELEMETRY.get().map_or(0, |telemetry| {
        telemetry
            .read()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .hits
            .load(Ordering::Relaxed)
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
    _timeout_seconds: u32,
) -> Result<ProbeOutcome, LinkError> {
    Err(LinkError::ProbeRequiresX86_64)
}

#[cfg(target_arch = "x86_64")]
extern "C" fn guest_exit(code: i32) {
    eprintln!("[REMu][PROBE] guest exit handler called with code {code}");
    run_all_guest_exit_callbacks();
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
    use std::collections::BTreeMap;
    use std::path::Path;
    use std::sync::{Arc, Mutex};

    use super::{
        AGC_IT_DISPATCH_DIRECT, AGC_IT_DISPATCH_INDIRECT, AGC_IT_INDEX_BASE,
        AGC_IT_INDEX_BUFFER_SIZE, AGC_IT_NOP, AGC_IT_NUM_INSTANCES, AGC_IT_SET_BASE,
        AGC_IT_SET_CONTEXT_REG_INDIRECT, AGC_IT_SET_SH_REG_INDIRECT, AGC_IT_SET_UCONFIG_REG_INDEX,
        AGC_IT_SET_UCONFIG_REG_INDIRECT, AGC_R_ACB_RESET, AGC_R_DRAW_INDEX_AUTO,
        AgcColorTargetState, AgcComputeDispatchError, AgcGraphicsState, AgcShaderRegistry,
        DEFAULT_THREAD_AFFINITY_MASK, GuestExitCallback, ProbeDispatchState, PthreadAttrState,
        SAVE_DATA_ERROR_EXISTS, SAVE_DATA_ERROR_NOT_FOUND, SAVE_DATA_PARAM_SIZE,
        SCE_KERNEL_ERROR_NOT_FOUND, SCE_KERNEL_ERROR_NOT_IMPLEMENTED, VideoOutBufferAttributes,
        VideoOutState, agc_apply_set_base, agc_apply_submission_registers, agc_color_target_state,
        agc_dcb_draw_index_auto, agc_dcb_set_index_buffer, agc_dcb_set_num_instances,
        agc_decode_compute_dispatch, agc_driver_submit_acb, agc_pm4,
        agc_submit_ordered_video_out_flip, allocate_direct_memory, allocate_pthread_thread_object,
        create_pthread_attr, decode_agc_memory_acquire, destroy_pthread_attr,
        dispatch_guest_numeric_parser, dispatch_probe_import, guest_clock_gettime, guest_free,
        guest_global_locale, guest_kernel_fstat, guest_kernel_ftruncate, guest_kernel_lseek,
        guest_kernel_read, guest_posix_file_result, guest_qsort, guest_tolower_table,
        guest_toupper_table, guest_wmemchr, guest_wmemcmp, guest_wmemcpy, guest_wmemmove,
        kernel_release_direct_memory, kernel_stat_payload, kernel_virtual_query, low_f32, low_u32,
        pthread_attr_get_affinity, pthread_attr_get_stack, pthread_attr_set_detach_state,
        pthread_attr_set_schedule_param, pthread_attr_set_stack_size, read_guest_u16,
        read_guest_u32, read_guest_u64, resolve_guest_path, sanitize_save_segment,
        save_data_commit, save_data_create_transaction_resource, save_data_dir_name_search,
        save_data_mount3, save_data_pattern_matches, save_data_prepare, save_data_save_icon,
        save_data_set_param, save_data_umount2, sce_kernel_error, set_current_guest_thread_pointer,
        system_service_param_get_int, take_exit_callbacks, video_out_open, write_guest_u32,
        write_guest_u64, write_import_stub,
    };

    #[repr(C, align(256))]
    struct AlignedComputeProgram([u32; 4]);

    #[repr(C, align(8))]
    struct AlignedIndirectArguments([u32; 8]);

    #[repr(C)]
    struct AgcSubmitPacket {
        address: u64,
        dword_count: u32,
        flags: [u8; 4],
    }

    static GPU_EVENT_SINK_TEST: Mutex<()> = Mutex::new(());

    fn compute_graphics_state(program_address: u64) -> (AgcGraphicsState, AgcShaderRegistry) {
        let mut graphics = AgcGraphicsState::default();
        for (offset, value) in [
            (
                remu_gpu::COMPUTE_PGM_LO_REGISTER,
                low_u32(program_address >> 8),
            ),
            (
                remu_gpu::COMPUTE_PGM_HI_REGISTER,
                ((program_address >> 40) & 0xff) as u32,
            ),
            (
                remu_gpu::COMPUTE_PGM_RSRC2_REGISTER,
                (2 << 1) | (1 << 7) | (1 << 9) | (1 << 10),
            ),
            (remu_gpu::COMPUTE_NUM_THREAD_X_REGISTER, 64),
            (remu_gpu::COMPUTE_NUM_THREAD_Y_REGISTER, 2),
            (remu_gpu::COMPUTE_NUM_THREAD_Z_REGISTER, 1),
            (0x240, 0x1234_5678),
            (0x241, 0x9abc_def0),
        ] {
            graphics.shader_registers.insert(offset, value);
        }
        let mut shaders = AgcShaderRegistry::default();
        shaders.snapshots.insert(
            program_address,
            remu_gpu::ShaderSnapshot {
                header_address: 0,
                program: Arc::new(remu_shader::Program::new(program_address, Vec::new())),
                metadata: None,
            },
        );
        (graphics, shaders)
    }

    fn targetless_scanout_test_draw() -> remu_gpu::IndexedDraw {
        let pixel_program = remu_shader::Program::new(
            0x2000,
            vec![remu_shader::Instruction {
                pc: 0,
                encoding: remu_shader::ShaderEncoding::Exp,
                opcode: remu_shader::Opcode::EXPORT,
                words: vec![0, 0],
                sources: Vec::new(),
                destinations: Vec::new(),
                control: Some(remu_shader::InstructionControl::Export(
                    remu_shader::ExportControl {
                        target: 0,
                        enable_mask: 0xf,
                        compressed: false,
                        done: true,
                        valid_mask: false,
                    },
                )),
            }],
        );
        remu_gpu::IndexedDraw {
            sequence: 1,
            indexed: false,
            instance_count: 1,
            index_buffer_address: 0,
            index_resource_address: 0,
            index_buffer_count: 0,
            index_size: 0,
            index_offset: 0,
            index_count: 3,
            draw_flags: 0,
            primitive_type: 4,
            export_shader_address: 0,
            pixel_shader_address: 0x2000,
            export_shader_prefix: [0; 4],
            pixel_shader_prefix: [0; 4],
            export_shader: None,
            pixel_shader: Some(remu_gpu::ShaderSnapshot {
                header_address: 0,
                program: Arc::new(pixel_program),
                metadata: None,
            }),
            export_shader_evaluation: None,
            pixel_shader_evaluation: None,
            color_target_address: 0,
            scanout_target: None,
            context_registers: vec![remu_gpu::GpuRegister {
                offset: super::AGC_CB_SHADER_MASK,
                value: 0xf,
            }],
            shader_registers: Vec::new(),
            user_config_registers: Vec::new(),
        }
    }

    unsafe extern "C" fn compare_i32(left: *const u8, right: *const u8) -> i32 {
        // SAFETY: guest_qsort passes pointers to complete i32 elements in the test array.
        let left = unsafe { std::ptr::read_unaligned(left.cast::<i32>()) };
        // SAFETY: guest_qsort passes pointers to complete i32 elements in the test array.
        let right = unsafe { std::ptr::read_unaligned(right.cast::<i32>()) };
        match left.cmp(&right) {
            std::cmp::Ordering::Less => -1,
            std::cmp::Ordering::Equal => 0,
            std::cmp::Ordering::Greater => 1,
        }
    }

    #[test]
    fn posix_file_errors_return_minus_one_and_store_guest_errno() {
        let mut tls = [0_u64; 16];
        let thread_pointer = u64::try_from(tls.as_mut_ptr().addr()).unwrap();
        set_current_guest_thread_pointer(thread_pointer);
        assert_eq!(guest_posix_file_result(7), 7);
        assert_eq!(read_guest_u32(thread_pointer + 0x40), 0);
        assert_eq!(
            guest_posix_file_result(super::sce_kernel_error(SCE_KERNEL_ERROR_NOT_FOUND)),
            u64::MAX
        );
        assert_eq!(
            read_guest_u32(thread_pointer + 0x40),
            libc::ENOENT.cast_unsigned()
        );
        assert_eq!(
            guest_posix_file_result(super::sce_kernel_error(
                super::SCE_KERNEL_ERROR_INVALID_ARGUMENT
            )),
            u64::MAX
        );
        assert_eq!(
            read_guest_u32(thread_pointer + 0x40),
            libc::EINVAL.cast_unsigned()
        );
        set_current_guest_thread_pointer(0);
    }

    #[test]
    fn video_out_open_owns_the_single_main_display_port() {
        let mut state = ProbeDispatchState::default();
        assert_eq!(video_out_open([0xff, 0, 0, 0, 0, 0], &mut state), 1);
        assert_eq!(state.video_out.user_id, 0xff);
        assert_eq!(state.video_out.bus_type, 0);
        assert_eq!(state.video_out.index, 0);
        assert_eq!(
            video_out_open([0xff, 0, 0, 0, 0, 0], &mut state),
            sce_kernel_error(super::VIDEO_OUT_ERROR_RESOURCE_BUSY)
        );

        let mut other = ProbeDispatchState::default();
        assert_eq!(
            video_out_open([0xff, 1, 0, 0, 0, 0], &mut other),
            sce_kernel_error(super::VIDEO_OUT_ERROR_INVALID_VALUE)
        );
    }

    #[test]
    fn copies_gen5_wide_characters_as_two_byte_elements() {
        let mut words = [
            0x41_u16, 0x42, 0x43, 0x44, 0x45, 0x46, 0x47, 0x48, 0xaaaa, 0xbbbb, 0xcccc, 0xdddd,
            0xeeee,
        ];
        let base = u64::try_from(words.as_mut_ptr().addr()).unwrap();
        let destination = base + 4 * 2;

        assert_eq!(guest_wmemcpy(destination, base, 8), destination);
        assert_eq!(
            words,
            [
                0x41, 0x42, 0x43, 0x44, 0x41, 0x42, 0x43, 0x44, 0x45, 0x46, 0x47, 0x48, 0xeeee,
            ]
        );

        assert_eq!(guest_wmemmove(base, destination, 4), base);
        assert_eq!(&words[..4], &[0x41, 0x42, 0x43, 0x44]);
    }

    #[test]
    fn searches_and_compares_gen5_wide_characters() {
        let left = [0x0061_u16, 0x0062, 0x20ac, 0x0064];
        let right = [0x0061_u16, 0x0062, 0x20ad, 0x0064];
        let left_address = u64::try_from(left.as_ptr().addr()).unwrap();
        let right_address = u64::try_from(right.as_ptr().addr()).unwrap();

        assert_eq!(guest_wmemchr(left_address, 0x20ac, 4), left_address + 4);
        assert_eq!(guest_wmemchr(left_address, 0x9999, 4), 0);
        assert_eq!(guest_wmemcmp(left_address, left_address, 4), 0);
        assert_eq!(guest_wmemcmp(left_address, right_address, 4), u64::MAX);
        assert_eq!(guest_wmemcmp(right_address, left_address, 4), 1);
    }

    #[test]
    fn exposes_a_stable_dinkumware_global_locale() {
        let mut state = ProbeDispatchState::default();
        let locale = guest_global_locale(&mut state);
        let vtable = read_guest_u64(locale);
        let retain = read_guest_u64(vtable + 0x10);
        // SAFETY: guest_global_locale installs this exact process-local C ABI function pointer.
        let retain: extern "C" fn(u64) -> u64 = unsafe {
            std::mem::transmute(usize::try_from(retain).expect("function pointer fits usize"))
        };

        assert_ne!(locale, 0);
        assert_ne!(vtable, 0);
        assert_eq!(retain(locale), locale);
        assert_eq!(guest_global_locale(&mut state), locale);

        guest_free(state.locale_object);
        guest_free(state.locale_vtable);
    }

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

    #[test]
    fn builds_the_gen5_kernel_stat_layout() {
        let payload = kernel_stat_payload(1_025, false, b"/app0/data.bin");

        assert_eq!(
            u16::from_le_bytes(payload[8..10].try_into().unwrap()),
            0x81ff
        );
        assert_eq!(u16::from_le_bytes(payload[10..12].try_into().unwrap()), 1);
        assert_eq!(
            i64::from_le_bytes(payload[72..80].try_into().unwrap()),
            1_025
        );
        assert_eq!(i64::from_le_bytes(payload[80..88].try_into().unwrap()), 3);
        assert_eq!(u32::from_le_bytes(payload[88..92].try_into().unwrap()), 512);
    }

    #[test]
    fn sign_extends_gen5_kernel_errors() {
        assert_eq!(sce_kernel_error(0x8002_0002), 0xffff_ffff_8002_0002);
    }

    #[test]
    fn translates_console_clock_ids_to_realtime() {
        let before = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs();
        let mut output = libc::timespec {
            tv_sec: 0,
            tv_nsec: 0,
        };
        assert_eq!(guest_clock_gettime(4, (&raw mut output).addr() as u64), 0);
        let seconds = u64::try_from(output.tv_sec).unwrap();
        assert!(seconds >= before);
        assert!(seconds <= before + 2);
        assert!((0..1_000_000_000).contains(&output.tv_nsec));
    }

    #[test]
    fn builds_distinct_dinkumware_case_mapping_tables() {
        let mut state = ProbeDispatchState::default();
        let lower = guest_tolower_table(&mut state);
        let upper = guest_toupper_table(&mut state);

        assert_ne!(lower, 0);
        assert_ne!(upper, 0);
        assert_ne!(lower, upper);
        assert_eq!(read_guest_u16(lower + u64::from(b'A') * 2), u16::from(b'a'));
        assert_eq!(read_guest_u16(lower + u64::from(b'a') * 2), u16::from(b'a'));
        assert_eq!(read_guest_u16(upper + u64::from(b'a') * 2), u16::from(b'A'));
        assert_eq!(read_guest_u16(upper + u64::from(b'A') * 2), u16::from(b'A'));
        assert_eq!(read_guest_u16(lower - 2), u16::MAX);
        assert_eq!(read_guest_u16(upper - 2), u16::MAX);

        guest_free(lower - 128 * 2);
        guest_free(upper - 128 * 2);
    }

    #[test]
    fn returns_stable_integer_system_parameters() {
        let mut value = 0_i32;
        let output = (&raw mut value).addr() as u64;

        assert_eq!(system_service_param_get_int([1, output, 0, 0, 0, 0]), 0);
        assert_eq!(value, 1);
        assert_eq!(system_service_param_get_int([4, output, 0, 0, 0, 0]), 0);
        assert_eq!(value, 180);
        assert_eq!(
            system_service_param_get_int([1, 0, 0, 0, 0, 0]),
            sce_kernel_error(0x80a1_0003)
        );
    }

    #[test]
    fn unknown_imports_return_the_gen5_not_implemented_error() {
        let mut state = ProbeDispatchState::default();
        assert_eq!(
            dispatch_probe_import("not-a-real-nid", [0; 6], [0; 6], &mut state),
            sce_kernel_error(SCE_KERNEL_ERROR_NOT_IMPLEMENTED)
        );
        assert_eq!(state.unimplemented_nids.len(), 1);
    }

    #[test]
    fn import_stubs_preserve_guest_rax_for_frame_capture() {
        let mut slot = [0_u8; super::IMPORT_STUB_SIZE_USIZE];
        write_import_stub(&mut slot, 0x1234, 0x1122_3344_5566_7788).unwrap();

        assert_eq!(slot[0], 0x68);
        assert_eq!(i32::from_le_bytes(slot[1..5].try_into().unwrap()), 0x1234);
        assert_eq!(&slot[5..7], &[0x49, 0xbb]);
        assert_eq!(
            u64::from_le_bytes(slot[7..15].try_into().unwrap()),
            0x1122_3344_5566_7788
        );
        assert_eq!(&slot[15..18], &[0x41, 0xff, 0xe3]);
        assert!(slot[18..].iter().all(|byte| *byte == 0));
    }

    #[cfg(all(unix, target_arch = "x86_64"))]
    #[test]
    fn import_trampoline_captures_the_complete_sysv_register_frame() {
        const PAGE_SIZE: usize = 0x4000;
        let mut slot = [0_u8; super::IMPORT_STUB_SIZE_USIZE];
        write_import_stub(
            &mut slot,
            usize::try_from(super::TEST_CAPTURE_DISPATCH_INDEX).unwrap(),
            super::import_dispatch_trampoline as *const () as usize as u64,
        )
        .unwrap();
        // SAFETY: This test owns one anonymous page and changes it from writable to executable
        // before calling the complete import stub copied into it.
        let mapping = unsafe {
            libc::mmap(
                std::ptr::null_mut(),
                PAGE_SIZE,
                libc::PROT_READ | libc::PROT_WRITE,
                libc::MAP_PRIVATE | libc::MAP_ANON,
                -1,
                0,
            )
        };
        assert_ne!(mapping, libc::MAP_FAILED);
        // SAFETY: mapping owns PAGE_SIZE writable bytes and slot fits at its start.
        unsafe { std::ptr::copy_nonoverlapping(slot.as_ptr(), mapping.cast::<u8>(), slot.len()) };
        // SAFETY: mapping and its exact size came from mmap above.
        assert_eq!(
            unsafe { libc::mprotect(mapping, PAGE_SIZE, libc::PROT_READ | libc::PROT_EXEC) },
            0
        );
        *super::TEST_CAPTURED_IMPORT_FRAME
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner) = None;

        // SAFETY: the page now contains a complete executable import stub targeting the real
        // trampoline, and invoke_test_import_stub establishes the guest SysV register state.
        let result = unsafe { super::invoke_test_import_stub(mapping.addr() as u64) };
        let frame = super::TEST_CAPTURED_IMPORT_FRAME
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .take()
            .expect("test dispatch captured its frame");
        // SAFETY: mapping and its exact size came from mmap and are no longer in use.
        assert_eq!(unsafe { libc::munmap(mapping, PAGE_SIZE) }, 0);

        assert_eq!(
            frame.integer_arguments,
            [0x11, 0x22, 0x33, 0x44, 0x55, 0x66]
        );
        assert_eq!(frame.rax, 0x0102_0304_0506_0708);
        assert_eq!(frame.dispatch_index, super::TEST_CAPTURE_DISPATCH_INDEX);
        assert_ne!(frame.guest_return_address, 0);
        assert_eq!(frame.guest_rsp & 0xf, 8);
        for (index, xmm) in frame.xmm_arguments.iter().enumerate() {
            let bytes = xmm.to_le_bytes();
            let low = u64::from_le_bytes(bytes[..8].try_into().unwrap());
            let high = u64::from_le_bytes(bytes[8..].try_into().unwrap());
            assert_eq!(low, u64::try_from(index + 1).unwrap() * 0x101);
            assert_eq!(high, 0);
        }
        assert_eq!(result.integer, frame.rax);
        assert_eq!(result.float.to_bits(), 0x808);
    }

    #[test]
    fn pthread_attributes_preserve_observable_thread_creation_values() {
        let mut state = ProbeDispatchState::default();
        let mut opaque_attribute = 0_u64;
        let attribute_address = (&raw mut opaque_attribute).addr() as u64;
        assert_eq!(
            create_pthread_attr(attribute_address, PthreadAttrState::default(), &mut state),
            0
        );
        assert_ne!(opaque_attribute, 0);
        assert_eq!(
            pthread_attr_set_detach_state([attribute_address, 1, 0, 0, 0, 0], &mut state),
            0
        );
        assert_eq!(
            pthread_attr_set_stack_size([attribute_address, 0x20_0000, 0, 0, 0, 0], &mut state),
            0
        );
        let priority = 512_i32;
        assert_eq!(
            pthread_attr_set_schedule_param(
                [
                    attribute_address,
                    (&raw const priority).addr() as u64,
                    0,
                    0,
                    0,
                    0,
                ],
                &mut state,
            ),
            0
        );
        let mut stack_address = u64::MAX;
        let mut stack_size = 0_u64;
        let mut affinity_mask = 0_u64;
        assert_eq!(
            pthread_attr_get_stack(
                [
                    attribute_address,
                    (&raw mut stack_address).addr() as u64,
                    (&raw mut stack_size).addr() as u64,
                    0,
                    0,
                    0,
                ],
                &state,
            ),
            0
        );
        let attributes = state.pthread_attributes[&attribute_address].attributes;
        assert_eq!(attributes.detach_state, 1);
        assert_eq!(attributes.stack_size, 0x20_0000);
        assert_eq!(attributes.schedule_priority, priority);
        assert_eq!(stack_address, 0);
        assert_eq!(stack_size, 0x20_0000);
        assert_eq!(
            pthread_attr_get_affinity(
                [
                    attribute_address,
                    (&raw mut affinity_mask).addr() as u64,
                    0,
                    0,
                    0,
                    0,
                ],
                &state,
            ),
            DEFAULT_THREAD_AFFINITY_MASK
        );
        assert_eq!(affinity_mask, DEFAULT_THREAD_AFFINITY_MASK);
        assert_eq!(destroy_pthread_attr(attribute_address, &mut state), 0);
        assert_eq!(opaque_attribute, 0);
    }

    #[test]
    fn pthread_threads_are_distinct_guest_pointer_objects() {
        let mut state = ProbeDispatchState {
            next_thread_unique_id: 1,
            ..ProbeDispatchState::default()
        };
        let first = allocate_pthread_thread_object(&mut state, "worker").unwrap();
        let second = allocate_pthread_thread_object(&mut state, "").unwrap();

        assert_ne!(first, 0);
        assert_ne!(first, second);
        assert_eq!(read_guest_u64(first), 0);
        assert_eq!(read_guest_u64(second + 0xff8), 0);
        assert_eq!(state.pthread_threads[&first].unique_id, 1);
        assert_eq!(state.pthread_threads[&first].name, "worker");
        assert_eq!(state.pthread_threads[&second].unique_id, 2);
        assert_eq!(state.pthread_threads[&second].name, "Thread-2");

        for object in state.pthread_threads.into_values() {
            guest_free(object.handle);
        }
    }

    #[test]
    fn finalization_selects_callbacks_in_reverse_registration_order() {
        let mut state = ProbeDispatchState {
            exit_callbacks: vec![
                GuestExitCallback {
                    function: 1,
                    argument: None,
                    module: 0,
                },
                GuestExitCallback {
                    function: 2,
                    argument: Some(20),
                    module: 7,
                },
                GuestExitCallback {
                    function: 3,
                    argument: Some(30),
                    module: 8,
                },
                GuestExitCallback {
                    function: 4,
                    argument: Some(40),
                    module: 7,
                },
            ],
            ..ProbeDispatchState::default()
        };
        let callbacks = take_exit_callbacks(&mut state, Some(7));
        assert_eq!(
            callbacks
                .iter()
                .map(|callback| callback.function)
                .collect::<Vec<_>>(),
            [4, 2]
        );
        assert_eq!(
            state
                .exit_callbacks
                .iter()
                .map(|callback| callback.function)
                .collect::<Vec<_>>(),
            [1, 3]
        );
    }

    #[test]
    fn decodes_float_argument_from_low_xmm_bits() {
        let expected = -1.25_f32;
        assert_eq!(
            low_f32(f64::from_bits(u64::from(expected.to_bits()))).to_bits(),
            expected.to_bits()
        );
    }

    #[test]
    fn parses_dinkumware_unsigned_longs_with_end_pointer() {
        let text = c"0x2a tail";
        let mut end = std::ptr::null_mut::<libc::c_char>();
        let result = dispatch_guest_numeric_parser(
            "zlfEH8FmyUA",
            [
                text.as_ptr().addr() as u64,
                (&raw mut end).addr() as u64,
                0,
                0,
                0,
                0,
            ],
        );

        assert_eq!(result.integer, 42);
        assert_eq!(end.addr() - text.as_ptr().addr(), 4);
    }

    #[test]
    fn sorts_guest_elements_with_the_guest_comparator() {
        let mut values = [7_i32, -2, 11, 0, 4];
        guest_qsort([
            u64::try_from(values.as_mut_ptr().addr()).unwrap(),
            u64::try_from(values.len()).unwrap(),
            u64::try_from(std::mem::size_of::<i32>()).unwrap(),
            u64::try_from(compare_i32 as *const () as usize).unwrap(),
            0,
            0,
        ]);
        assert_eq!(values, [-2, 0, 4, 7, 11]);
    }

    #[test]
    fn sanitizes_save_slot_path_segments() {
        assert_eq!(sanitize_save_segment("SAVEDATA00"), "SAVEDATA00");
        assert_eq!(sanitize_save_segment("../outside"), "___outside");
        assert_eq!(sanitize_save_segment("///"), "default");
    }

    #[test]
    fn matches_save_data_wildcards_case_insensitively() {
        assert!(save_data_pattern_matches("SAVEDATA00", "save%"));
        assert!(save_data_pattern_matches("slot-A", "SLOT__"));
        assert!(save_data_pattern_matches("alpha-beta", "%ha_b%"));
        assert!(!save_data_pattern_matches("slot", "slot_"));
        assert!(!save_data_pattern_matches("other", "save%"));
    }

    #[test]
    #[allow(clippy::too_many_lines)]
    fn persists_and_searches_transactional_save_data() {
        let temporary = TemporaryTestDirectory::new("savedata-transaction");
        let mut state = ProbeDispatchState {
            save_root: temporary.path.clone(),
            save_data_initialized: true,
            next_save_transaction_resource: 1,
            ..ProbeDispatchState::default()
        };
        std::fs::create_dir_all(&state.save_root).unwrap();

        let resource = save_data_create_transaction_resource([0xc_0000, 0, 0, 0, 0, 0], &mut state);
        assert_eq!(resource, 1);
        let resource = u32::try_from(resource).unwrap();
        let mut dir_name = [0_u8; 32];
        dir_name[..10].copy_from_slice(b"SAVEDATA00");
        let mut mount = [0_u8; 0x30];
        let mut mount_result = [0_u8; 0x40];
        let mount_address = mount.as_mut_ptr().addr() as u64;
        write_guest_u32(mount_address, 0x1000_0000);
        write_guest_u64(mount_address + 0x08, dir_name.as_ptr().addr() as u64);
        write_guest_u64(mount_address + 0x10, 56);
        write_guest_u32(mount_address + 0x20, 1 << 5);
        write_guest_u32(mount_address + 0x28, resource);
        let result_address = mount_result.as_mut_ptr().addr() as u64;
        assert_eq!(
            save_data_mount3([mount_address, result_address, 0, 0, 0, 0], &mut state),
            0
        );
        assert_eq!(&mount_result[..11], b"/savedata0\0");
        assert_eq!(
            u32::from_le_bytes(mount_result[0x1c..0x20].try_into().unwrap()),
            1
        );

        let mut prepare = [0_u8; 0x28];
        prepare[..4].copy_from_slice(&resource.to_le_bytes());
        assert_eq!(
            save_data_prepare(
                [result_address, prepare.as_ptr().addr() as u64, 0, 0, 0, 0,],
                &mut state,
            ),
            0
        );

        let mut param = [0_u8; SAVE_DATA_PARAM_SIZE];
        param[..14].copy_from_slice(b"Save Metadata\0");
        param[0x500..0x504].copy_from_slice(&42_u32.to_le_bytes());
        assert_eq!(
            save_data_set_param(
                [
                    result_address,
                    0,
                    param.as_ptr().addr() as u64,
                    SAVE_DATA_PARAM_SIZE as u64,
                    0,
                    0,
                ],
                &mut state,
            ),
            0
        );
        let icon_bytes = b"\x89PNG\r\n\x1a\ntransaction-icon";
        let icon = [
            icon_bytes.as_ptr().addr() as u64,
            icon_bytes.len() as u64,
            icon_bytes.len() as u64,
        ];
        assert_eq!(
            save_data_save_icon(
                [result_address, icon.as_ptr().addr() as u64, 0, 0, 0, 0],
                &state,
            ),
            0
        );

        let committed_param = state.save_root.join("SAVEDATA00/sce_sys/param.bin");
        assert_ne!(
            &std::fs::read(&committed_param).unwrap()[..14],
            b"Save Metadata\0"
        );
        let mut commit = [0_u8; 0x28];
        commit[..4].copy_from_slice(&resource.to_le_bytes());
        assert_eq!(
            save_data_commit([commit.as_ptr().addr() as u64, 0, 0, 0, 0, 0], &mut state,),
            0
        );
        assert_eq!(
            &std::fs::read(&committed_param).unwrap()[..14],
            b"Save Metadata\0"
        );
        assert_eq!(
            std::fs::read(state.save_root.join("SAVEDATA00/sce_sys/icon0.png")).unwrap(),
            icon_bytes
        );

        let mut search_condition = [0_u8; 0x40];
        let mut search_result = [0_u8; 0x30];
        let pattern = b"save%\0";
        let mut found_names = [0_u8; 64];
        let mut found_params = vec![0_u8; SAVE_DATA_PARAM_SIZE * 2];
        let mut found_infos = [0_u8; 0x60];
        let condition_address = search_condition.as_mut_ptr().addr() as u64;
        let search_result_address = search_result.as_mut_ptr().addr() as u64;
        write_guest_u32(condition_address, 0x1000_0000);
        write_guest_u64(condition_address + 0x10, pattern.as_ptr().addr() as u64);
        write_guest_u32(condition_address + 0x18, 1);
        write_guest_u64(
            search_result_address + 0x08,
            found_names.as_mut_ptr().addr() as u64,
        );
        write_guest_u32(search_result_address + 0x10, 2);
        write_guest_u64(
            search_result_address + 0x18,
            found_params.as_mut_ptr().addr() as u64,
        );
        write_guest_u64(
            search_result_address + 0x20,
            found_infos.as_mut_ptr().addr() as u64,
        );
        assert_eq!(
            save_data_dir_name_search(
                [condition_address, search_result_address, 0, 0, 0, 0],
                &state,
            ),
            0
        );
        assert_eq!(
            u32::from_le_bytes(search_result[..4].try_into().unwrap()),
            1
        );
        assert_eq!(
            u32::from_le_bytes(search_result[0x14..0x18].try_into().unwrap()),
            1
        );
        assert_eq!(&found_names[..11], b"SAVEDATA00\0");
        assert_eq!(&found_params[..14], b"Save Metadata\0");
        assert_eq!(u64::from_le_bytes(found_infos[..8].try_into().unwrap()), 56);

        assert_eq!(
            save_data_umount2([0, result_address, 0, 0, 0, 0], &mut state),
            0
        );
        mount_result.fill(0);
        assert_eq!(
            save_data_mount3([mount_address, result_address, 0, 0, 0, 0], &mut state),
            0
        );
        assert_eq!(
            u32::from_le_bytes(mount_result[0x1c..0x20].try_into().unwrap()),
            0
        );
        assert_eq!(
            save_data_umount2([0, result_address, 0, 0, 0, 0], &mut state),
            0
        );
        write_guest_u32(mount_address + 0x20, 1 << 2);
        assert_eq!(
            save_data_mount3([mount_address, result_address, 0, 0, 0, 0], &mut state),
            sce_kernel_error(SAVE_DATA_ERROR_EXISTS)
        );
        dir_name.fill(0);
        dir_name[..7].copy_from_slice(b"MISSING");
        write_guest_u32(mount_address + 0x20, 1 << 1);
        assert_eq!(
            save_data_mount3([mount_address, result_address, 0, 0, 0, 0], &mut state),
            sce_kernel_error(SAVE_DATA_ERROR_NOT_FOUND)
        );
    }

    struct TemporaryTestDirectory {
        path: std::path::PathBuf,
    }

    impl TemporaryTestDirectory {
        fn new(label: &str) -> Self {
            static NEXT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(1);
            let id = NEXT.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            let path =
                std::env::temp_dir().join(format!("remu-{label}-{}-{id}", std::process::id()));
            if path.exists() {
                std::fs::remove_dir_all(&path).unwrap();
            }
            std::fs::create_dir(&path).unwrap();
            Self { path }
        }
    }

    impl Drop for TemporaryTestDirectory {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.path);
        }
    }

    #[test]
    fn implements_seek_read_stat_and_truncate_on_one_guest_descriptor() {
        let temporary = TemporaryTestDirectory::new("file-abi");
        let path = temporary.path.join("payload.bin");
        std::fs::write(&path, b"abcdef").unwrap();
        let file = std::fs::OpenOptions::new()
            .read(true)
            .write(true)
            .open(path)
            .unwrap();
        let mut state = ProbeDispatchState::default();
        state.kernel_files.insert(3, file);

        assert_eq!(guest_kernel_lseek([3, 2, 0, 0, 0, 0], &mut state), 2);
        let mut bytes = [0_u8; 2];
        assert_eq!(
            guest_kernel_read(
                [3, bytes.as_mut_ptr().addr() as u64, 2, 0, 0, 0],
                &mut state
            ),
            2
        );
        assert_eq!(&bytes, b"cd");

        let mut stat = [0_u8; super::SCE_KERNEL_STAT_SIZE];
        assert_eq!(
            guest_kernel_fstat([3, stat.as_mut_ptr().addr() as u64, 0, 0, 0, 0], &state),
            0
        );
        assert_eq!(i64::from_le_bytes(stat[72..80].try_into().unwrap()), 6);
        assert_ne!(u32::from_le_bytes(stat[4..8].try_into().unwrap()), 0);
        assert_ne!(u16::from_le_bytes(stat[8..10].try_into().unwrap()), 0);
        assert_ne!(u32::from_le_bytes(stat[88..92].try_into().unwrap()), 0);

        assert_eq!(guest_kernel_ftruncate([3, 4, 0, 0, 0, 0], &mut state), 0);
        assert_eq!(
            state
                .kernel_files
                .get(&3)
                .unwrap()
                .metadata()
                .unwrap()
                .len(),
            4
        );
    }

    #[cfg(target_os = "macos")]
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
            super::map_direct_memory(
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
        assert_eq!(output[32], 0x12);
        assert_eq!(&output[33..39], b"direct");
    }

    #[cfg(target_os = "macos")]
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

    #[cfg(target_os = "macos")]
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

    #[cfg(target_os = "macos")]
    #[test]
    fn direct_memory_mappings_alias_the_same_physical_bytes() {
        use super::{kernel_munmap, map_direct_memory};

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

    #[cfg(target_os = "macos")]
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
            super::map_direct_memory(
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
            super::map_direct_memory(
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
            super::map_direct_memory(
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
            super::kernel_munmap([fixed, 0x4000, 0, 0, 0, 0], &mut state),
            0
        );
        assert_eq!(
            super::kernel_munmap([hinted, 0x4000, 0, 0, 0, 0], &mut state),
            0
        );
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn gen5_direct_memory_v2_and_checked_release_preserve_their_distinct_abi() {
        let mut state = ProbeDispatchState::default();
        assert_eq!(
            super::kernel_checked_release_direct_memory([0, 0x4000, 0, 0, 0, 0], &mut state),
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
            super::map_direct_memory_2(
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
            super::kernel_checked_release_direct_memory([direct, 0x4000, 0, 0, 0, 0], &mut state),
            0
        );
        assert_eq!(
            super::kernel_checked_release_direct_memory([direct, 0x4000, 0, 0, 0, 0], &mut state),
            sce_kernel_error(SCE_KERNEL_ERROR_NOT_FOUND)
        );
        assert_eq!(
            super::kernel_release_direct_memory([direct, 0x4000, 0, 0, 0, 0], &mut state),
            0
        );
        assert_eq!(
            super::kernel_munmap([mapped, 0x4000, 0, 0, 0, 0], &mut state),
            0
        );
    }

    #[test]
    fn decodes_gen5_memory_acquire_ranges_and_controls() {
        let packet = [
            agc_pm4(8, AGC_IT_NOP, super::AGC_R_ACQUIRE_MEM),
            0x8000_1234,
            0x3456_789a,
            0x0000_00ab,
            0x7654_3210,
            0x0000_cdef,
            0xabcd_2468,
            0xfff5_4321,
        ];
        let decoded = decode_agc_memory_acquire(packet.as_ptr().addr() as u64);

        assert_eq!(decoded.engine, 1);
        assert_eq!(decoded.cb_db_control, 0x1234);
        assert_eq!(decoded.base_address, 0x00cd_ef76_5432_1000);
        assert_eq!(decoded.size_bytes, 0xab34_5678_9a00);
        assert_eq!(decoded.poll_interval, 0x2468);
        assert_eq!(decoded.gcr_control, 0x5_4321);
    }

    #[test]
    fn applies_native_gen5_indirect_register_packets_to_each_register_space() {
        let context_registers = [0x318, 0x1234_5678, 0x390, 0x5a, u32::MAX, 0xfeed_face];
        let shader_registers = [
            0x2000_020c,
            0x7654_3210,
            0x2000_020d,
            0x44,
            0x2000_0213,
            0x388,
        ];
        let user_config_registers = [0x4000_0242, 4];
        let packet = |opcode, registers: &[u32]| {
            let address = registers.as_ptr().addr() as u64;
            [
                agc_pm4(5, opcode, 0),
                low_u32(address),
                (address >> 32) as u32,
                0x8000_0000,
                u32::try_from(registers.len() / 2).expect("register pair count"),
            ]
        };
        let mut graphics = AgcGraphicsState::default();

        for (opcode, registers) in [
            (AGC_IT_SET_CONTEXT_REG_INDIRECT, &context_registers[..]),
            (AGC_IT_SET_SH_REG_INDIRECT, &shader_registers[..]),
            (AGC_IT_SET_UCONFIG_REG_INDIRECT, &user_config_registers[..]),
        ] {
            let command = packet(opcode, registers);
            agc_apply_submission_registers(
                command.as_ptr().addr() as u64,
                5,
                opcode,
                0,
                &mut graphics,
            )
            .expect("native indirect register packet");
        }

        assert_eq!(graphics.context_registers.get(&0x318), Some(&0x1234_5678));
        assert_eq!(graphics.context_registers.get(&0x390), Some(&0x5a));
        assert!(!graphics.context_registers.contains_key(&u32::MAX));
        assert_eq!(
            graphics
                .shader_registers
                .get(&remu_gpu::COMPUTE_PGM_LO_REGISTER),
            Some(&0x7654_3210)
        );
        assert_eq!(
            graphics
                .shader_registers
                .get(&remu_gpu::COMPUTE_PGM_HI_REGISTER),
            Some(&0x44)
        );
        assert_eq!(graphics.user_config_registers.get(&0x242), Some(&4));
    }

    #[test]
    fn applies_native_gen5_indexed_uconfig_packet() {
        let packet = [
            agc_pm4(3, AGC_IT_SET_UCONFIG_REG_INDEX, 0),
            0x2000_0243,
            0x400 | 1 | (2 << 6),
        ];
        let mut graphics = AgcGraphicsState::default();

        agc_apply_submission_registers(
            packet.as_ptr().addr() as u64,
            3,
            AGC_IT_SET_UCONFIG_REG_INDEX,
            0,
            &mut graphics,
        )
        .expect("native indexed user-config register packet");

        assert_eq!(graphics.user_config_registers.get(&0x243), Some(&0x481));
        assert_eq!(graphics.index_size, 1);
    }

    #[test]
    fn clear_state_resets_complete_queue_state_without_discarding_residency() {
        let address = 0x0070_1234_5600_u64;
        let target = remu_gpu::GpuColorTarget {
            slot: 0,
            address,
            view: 0,
            info: 10 << 2,
            attrib: 0,
            attrib2: ((1_280 - 1) << 14) | (720 - 1),
            attrib3: (27 << 14) | (1 << 24),
        };
        let packet = [agc_pm4(2, super::AGC_IT_CLEAR_STATE, 0), 0x0f];
        let mut graphics = AgcGraphicsState::default();
        graphics.context_registers.insert(0x318, 1);
        graphics.shader_registers.insert(0x20c, 2);
        graphics.user_config_registers.insert(0x242, 3);
        graphics.index_buffer_address = 0x1000;
        graphics.index_resource_address = 0x2000;
        graphics.index_buffer_count = 9;
        graphics.index_size = 1;
        graphics.instance_count = 7;
        graphics.draw_index_offset = 5;
        graphics.compute_indirect_args_address = 0x3000;
        graphics.emitted_draws = 11;
        graphics.known_color_targets.insert(address, target);
        graphics.pending_targetless_draw = Some(Box::new(targetless_scanout_test_draw()));

        super::agc_handle_submission_packet(
            super::AgcSubmissionPacket {
                address: packet.as_ptr().addr() as u64,
                length: 2,
                opcode: super::AGC_IT_CLEAR_STATE,
                register: 0,
            },
            &mut graphics,
            &mut AgcShaderRegistry::default(),
            &BTreeMap::new(),
            &VideoOutState::default(),
            &mut Vec::new(),
        )
        .expect("valid CLEAR_STATE packet");

        assert!(graphics.context_registers.is_empty());
        assert!(graphics.shader_registers.is_empty());
        assert!(graphics.user_config_registers.is_empty());
        assert_eq!(graphics.index_buffer_address, 0);
        assert_eq!(graphics.index_resource_address, 0);
        assert_eq!(graphics.index_buffer_count, 0);
        assert_eq!(graphics.index_size, 0);
        assert_eq!(graphics.instance_count, 1);
        assert_eq!(graphics.draw_index_offset, 0);
        assert_eq!(graphics.compute_indirect_args_address, 0);
        assert!(graphics.pending_targetless_draw.is_none());
        assert_eq!(graphics.emitted_draws, 11);
        assert_eq!(graphics.known_color_targets.get(&address), Some(&target));
    }

    #[test]
    fn clear_state_rejects_non_native_layouts() {
        let invalid_control = [agc_pm4(2, super::AGC_IT_CLEAR_STATE, 0), 0x10];
        let mut graphics = AgcGraphicsState::default();
        let mut shaders = AgcShaderRegistry::default();
        let index_resources = BTreeMap::new();
        let video_out = VideoOutState::default();
        let mut flips = Vec::new();

        let mut invoke = |address, length, graphics: &mut AgcGraphicsState| {
            super::agc_handle_submission_packet(
                super::AgcSubmissionPacket {
                    address,
                    length,
                    opcode: super::AGC_IT_CLEAR_STATE,
                    register: 0,
                },
                graphics,
                &mut shaders,
                &index_resources,
                &video_out,
                &mut flips,
            )
        };
        assert!(matches!(
            invoke(invalid_control.as_ptr().addr() as u64, 2, &mut graphics),
            Err(super::AgcSubmissionError::UnsupportedClearStateControl(
                0x10
            ))
        ));
        assert!(matches!(
            invoke(invalid_control.as_ptr().addr() as u64, 3, &mut graphics),
            Err(super::AgcSubmissionError::ClearStatePacketLength { actual: 3 })
        ));
    }

    #[test]
    fn distinguishes_absent_programmed_and_malformed_color_target_state() {
        let address = 0x0070_1234_5600_u64;
        let mut registers = BTreeMap::from([
            (super::AGC_CB_COLOR0_BASE, low_u32(address >> 8)),
            (super::AGC_CB_COLOR0_BASE_EXT, low_u32(address >> 40)),
            (super::AGC_CB_COLOR0_VIEW, 0),
            (super::AGC_CB_COLOR0_INFO, 10 << 2),
            (super::AGC_CB_COLOR0_ATTRIB, 0),
            (
                super::AGC_CB_COLOR0_ATTRIB2,
                ((1_280 - 1) << 14) | (720 - 1),
            ),
            (super::AGC_CB_COLOR0_ATTRIB3, (27 << 14) | (1 << 24)),
        ]);

        let AgcColorTargetState::Programmed(targets) = agc_color_target_state(&registers) else {
            panic!("a complete target remains enabled when CB_TARGET_MASK is absent");
        };
        assert_eq!(targets.len(), 1);
        assert_eq!(targets[0].address, address);

        registers.remove(&super::AGC_CB_COLOR0_VIEW);
        assert!(matches!(
            agc_color_target_state(&registers),
            AgcColorTargetState::Malformed { slot: 0 }
        ));

        registers.insert(super::AGC_CB_TARGET_MASK, 0);
        assert!(matches!(
            agc_color_target_state(&registers),
            AgcColorTargetState::Absent
        ));
    }

    #[test]
    fn emits_deferred_scanout_draw_immediately_before_its_flip() {
        let _event_guard = GPU_EVENT_SINK_TEST
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let address = 0x0070_1234_5600_u64;
        let target = remu_gpu::GpuColorTarget {
            slot: 0,
            address,
            view: 0,
            info: 10 << 2,
            attrib: 0,
            attrib2: ((1_280 - 1) << 14) | (720 - 1),
            attrib3: (27 << 14) | (1 << 24),
        };
        let mut graphics = AgcGraphicsState::default();
        graphics.known_color_targets.insert(address, target);
        graphics.pending_targetless_draw = Some(Box::new(targetless_scanout_test_draw()));
        let video_out = VideoOutState {
            is_open: true,
            width: 1_280,
            height: 720,
            display_buffers: vec![address],
            display_buffer_attributes: vec![Some(VideoOutBufferAttributes {
                width: 1_280,
                height: 720,
                pitch: 1_280,
                pixel_format: 0x8000_0000_2200_0000,
                tiling_mode: 0,
            })],
            ..VideoOutState::default()
        };
        let (sender, receiver) = std::sync::mpsc::channel();
        remu_gpu::install_event_sink(sender);

        assert!(agc_submit_ordered_video_out_flip(
            &mut graphics,
            &video_out,
            1,
            0,
            0x1234,
        ));
        remu_gpu::clear_event_sink();

        let remu_gpu::GpuEvent::IndexedDraw(draw) = receiver.recv().expect("deferred draw event")
        else {
            panic!("draw must precede flip");
        };
        assert_eq!(
            draw.scanout_target,
            Some(remu_gpu::GpuScanoutTarget {
                address,
                width: 1_280,
                height: 720,
                pitch: 1_280,
                pixel_format: 0x8000_0000_2200_0000,
                tiling_mode: 0,
                export_slot: 0,
                write_mask: 0xf,
                color_surface: Some(target),
            })
        );
        let remu_gpu::GpuEvent::Flip {
            address: flipped_address,
            width,
            height,
            ..
        } = receiver.recv().expect("ordered flip event")
        else {
            panic!("flip must follow draw");
        };
        assert_eq!(flipped_address, address);
        assert_eq!((width, height), (1_280, 720));
        assert!(receiver.try_recv().is_err());
    }

    #[test]
    fn submits_descriptor_wrapped_acb_work_on_an_owner_scoped_compute_queue() {
        let _event_guard = GPU_EVENT_SINK_TEST
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let program = AlignedComputeProgram([0, 0, 0, 0]);
        let program_address = program.0.as_ptr().addr() as u64;
        let (_, shaders) = compute_graphics_state(program_address);
        let register_pairs = [
            remu_gpu::COMPUTE_PGM_LO_REGISTER,
            low_u32(program_address >> 8),
            remu_gpu::COMPUTE_PGM_HI_REGISTER,
            ((program_address >> 40) & 0xff) as u32,
            remu_gpu::COMPUTE_PGM_RSRC2_REGISTER,
            0,
            remu_gpu::COMPUTE_NUM_THREAD_X_REGISTER,
            64,
            remu_gpu::COMPUTE_NUM_THREAD_Y_REGISTER,
            1,
            remu_gpu::COMPUTE_NUM_THREAD_Z_REGISTER,
            1,
        ];
        let registers_address = register_pairs.as_ptr().addr() as u64;
        let commands = [
            agc_pm4(2, AGC_IT_NOP, AGC_R_ACB_RESET),
            0,
            agc_pm4(5, AGC_IT_SET_SH_REG_INDIRECT, 0),
            low_u32(registers_address),
            (registers_address >> 32) as u32,
            0x8000_0000,
            6,
            agc_pm4(5, AGC_IT_DISPATCH_DIRECT, 0),
            3,
            2,
            1,
            0x8045,
        ];
        let command_address = commands.as_ptr().addr() as u64;
        let descriptor = [
            low_u32(command_address),
            (command_address >> 32) as u32,
            u32::try_from(commands.len()).expect("command dword count"),
            0,
            0x5533_ccaa,
        ];
        let packet = AgcSubmitPacket {
            address: descriptor.as_ptr().addr() as u64,
            dword_count: 5,
            flags: [0; 4],
        };
        let (sender, receiver) = std::sync::mpsc::channel();
        remu_gpu::install_event_sink(sender);
        let mut state = ProbeDispatchState {
            agc_shader_registry: shaders,
            ..ProbeDispatchState::default()
        };

        let result = agc_driver_submit_acb(
            [7, (&raw const packet).addr() as u64, 0, 0, 0, 0],
            &mut state,
        );
        let event = receiver.recv_timeout(std::time::Duration::from_secs(1));
        remu_gpu::clear_event_sink();

        assert_eq!(result, 0);
        let remu_gpu::GpuEvent::ComputeDispatch(dispatch) = event.expect("compute dispatch event")
        else {
            panic!("ACB emitted a non-compute event");
        };
        assert_eq!(dispatch.source, remu_gpu::ComputeDispatchSource::Direct);
        assert_eq!(
            dispatch.launch.grid,
            remu_gpu::ComputeGrid::Threadgroups([3, 2, 1])
        );
        assert_eq!(dispatch.shader_address, program_address);
        assert!(state.agc_graphics.shader_registers.is_empty());
        assert_eq!(
            state.agc_compute_queues.get(&7).and_then(|queue| queue
                .shader_registers
                .get(&remu_gpu::COMPUTE_PGM_LO_REGISTER)),
            Some(&low_u32(program_address >> 8))
        );
    }

    #[test]
    fn decodes_and_evaluates_a_complete_direct_compute_dispatch_snapshot() {
        let program = AlignedComputeProgram([0xbeef_cafe, 0x1234_5678, 0, 0]);
        let program_address = program.0.as_ptr().addr() as u64;
        let (mut graphics, mut shaders) = compute_graphics_state(program_address);
        let packet = [agc_pm4(5, AGC_IT_DISPATCH_DIRECT, 0), 10, 4, 2, 0x8045];

        let dispatch = agc_decode_compute_dispatch(
            packet.as_ptr().addr() as u64,
            5,
            AGC_IT_DISPATCH_DIRECT,
            &mut graphics,
            &mut shaders,
        )
        .expect("complete direct compute dispatch");

        assert_eq!(dispatch.sequence, 1);
        assert_eq!(dispatch.source, remu_gpu::ComputeDispatchSource::Direct);
        assert_eq!(
            dispatch.launch.grid,
            remu_gpu::ComputeGrid::Threadgroups([10, 4, 2])
        );
        assert_eq!(dispatch.launch.threads_per_group, [64, 2, 1]);
        assert_eq!(dispatch.launch.wave_lane_count, 32);
        assert_eq!(dispatch.shader_address, program_address);
        assert_eq!(dispatch.shader_prefix, program.0);
        let evaluation = dispatch
            .shader_evaluation
            .as_ref()
            .expect("compute evaluation");
        assert_eq!(evaluation.initial_scalar_registers[0], 0x1234_5678);
        assert_eq!(evaluation.initial_scalar_registers[1], 0x9abc_def0);
        let system = evaluation
            .compute_system_registers
            .expect("compute system registers");
        assert_eq!(system.work_group_x, Some(2));
        assert_eq!(system.work_group_y, None);
        assert_eq!(system.work_group_z, Some(3));
        assert_eq!(system.thread_group_size, Some(4));
    }

    #[test]
    fn resolves_base_relative_and_absolute_compute_indirect_dimensions() {
        let program = AlignedComputeProgram([0, 0, 0, 0]);
        let program_address = program.0.as_ptr().addr() as u64;
        let (mut graphics, mut shaders) = compute_graphics_state(program_address);
        let arguments = AlignedIndirectArguments([0, 0, 7, 3, 1, 11, 5, 2]);
        let arguments_address = arguments.0.as_ptr().addr() as u64;
        let set_base = [
            agc_pm4(4, AGC_IT_SET_BASE, 0) | (1 << 1),
            1,
            low_u32(arguments_address),
            (arguments_address >> 32) as u32,
        ];
        agc_apply_set_base(set_base.as_ptr().addr() as u64, 4, &mut graphics)
            .expect("compute indirect base");
        assert_eq!(graphics.compute_indirect_args_address, arguments_address);

        let relative = [agc_pm4(3, AGC_IT_DISPATCH_INDIRECT, 0), 8, 0x8041];
        let relative_dispatch = agc_decode_compute_dispatch(
            relative.as_ptr().addr() as u64,
            3,
            AGC_IT_DISPATCH_INDIRECT,
            &mut graphics,
            &mut shaders,
        )
        .expect("base-relative compute dispatch");
        assert_eq!(
            relative_dispatch.source,
            remu_gpu::ComputeDispatchSource::Indirect {
                dimensions_address: arguments_address + 8,
            }
        );
        assert_eq!(
            relative_dispatch.launch.grid,
            remu_gpu::ComputeGrid::Threadgroups([7, 3, 1])
        );

        let absolute_address = arguments_address + 20;
        let absolute = [
            agc_pm4(4, AGC_IT_DISPATCH_INDIRECT, 0),
            low_u32(absolute_address),
            (absolute_address >> 32) as u32,
            0x8041,
        ];
        let absolute_dispatch = agc_decode_compute_dispatch(
            absolute.as_ptr().addr() as u64,
            4,
            AGC_IT_DISPATCH_INDIRECT,
            &mut graphics,
            &mut shaders,
        )
        .expect("absolute compute dispatch");
        assert_eq!(
            absolute_dispatch.source,
            remu_gpu::ComputeDispatchSource::Indirect {
                dimensions_address: absolute_address,
            }
        );
        assert_eq!(
            absolute_dispatch.launch.grid,
            remu_gpu::ComputeGrid::Threadgroups([11, 5, 2])
        );
    }

    #[test]
    fn rejects_indirect_compute_without_a_compute_set_base_packet() {
        let program = AlignedComputeProgram([0; 4]);
        let (mut graphics, mut shaders) = compute_graphics_state(program.0.as_ptr().addr() as u64);
        let packet = [agc_pm4(3, AGC_IT_DISPATCH_INDIRECT, 0), 0x80, 0x8041];
        assert!(matches!(
            agc_decode_compute_dispatch(
                packet.as_ptr().addr() as u64,
                3,
                AGC_IT_DISPATCH_INDIRECT,
                &mut graphics,
                &mut shaders,
            ),
            Err(AgcComputeDispatchError::MissingIndirectBase)
        ));
    }

    #[test]
    fn builds_the_four_argument_gen5_index_buffer_packets() {
        let mut command_buffer = vec![0_u8; 0x40];
        let mut commands = vec![0_u32; 16];
        let indices = [0_u32, 1, 2, 1, 2, 3];
        let command_buffer_address =
            u64::try_from(command_buffer.as_mut_ptr().addr()).expect("command buffer address");
        let commands_address =
            u64::try_from(commands.as_mut_ptr().addr()).expect("command storage address");
        let index_buffer_address =
            u64::try_from(indices.as_ptr().addr()).expect("index buffer address");
        let index_resource_address = 0x70_0007_3030;
        write_guest_u64(command_buffer_address + 0x10, commands_address);
        write_guest_u64(command_buffer_address + 0x18, commands_address + 64);
        write_guest_u32(command_buffer_address + 0x30, 0);

        let mut state = ProbeDispatchState::default();
        let command = agc_dcb_set_index_buffer(
            [
                command_buffer_address,
                index_buffer_address,
                index_resource_address,
                16,
                0,
                0,
            ],
            &mut state,
        );

        assert_eq!(command, commands_address);
        assert_eq!(commands[0], agc_pm4(3, AGC_IT_INDEX_BASE, 0));
        assert_eq!(commands[1], low_u32(index_buffer_address));
        assert_eq!(commands[2], low_u32(index_buffer_address >> 32));
        assert_eq!(commands[3], agc_pm4(2, AGC_IT_INDEX_BUFFER_SIZE, 0));
        assert_eq!(commands[4], 16);
        assert_eq!(
            read_guest_u64(command_buffer_address + 0x10),
            commands_address + 20
        );
        assert_eq!(
            state.agc_index_resources.get(&index_buffer_address),
            Some(&index_resource_address)
        );
    }

    #[test]
    fn builds_the_gen5_instanced_auto_draw_packets() {
        let mut command_buffer = vec![0_u8; 0x40];
        let mut commands = vec![0_u32; 16];
        let command_buffer_address =
            u64::try_from(command_buffer.as_mut_ptr().addr()).expect("command buffer address");
        let commands_address =
            u64::try_from(commands.as_mut_ptr().addr()).expect("command storage address");
        write_guest_u64(command_buffer_address + 0x10, commands_address);
        write_guest_u64(command_buffer_address + 0x18, commands_address + 64);
        write_guest_u32(command_buffer_address + 0x30, 0);

        let instances = agc_dcb_set_num_instances([command_buffer_address, 3, 0, 0, 0, 0]);
        let draw = agc_dcb_draw_index_auto([command_buffer_address, 3, 0x4000_0000, 0, 0, 0]);

        assert_eq!(instances, commands_address);
        assert_eq!(commands[0], agc_pm4(2, AGC_IT_NUM_INSTANCES, 0));
        assert_eq!(commands[1], 3);
        assert_eq!(draw, commands_address + 8);
        assert_eq!(commands[2], agc_pm4(7, AGC_IT_NOP, AGC_R_DRAW_INDEX_AUTO));
        assert_eq!(commands[3], 3);
        assert_eq!(&commands[4..9], &[0; 5]);
        assert_eq!(
            read_guest_u64(command_buffer_address + 0x10),
            commands_address + 36
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
