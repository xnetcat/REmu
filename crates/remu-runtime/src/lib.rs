// Copyright (C) 2026 REmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

mod agc_command_abi;
mod agc_register_defaults;
mod audio;
#[cfg(target_os = "macos")]
mod kernel_memory;
mod np;

use std::cell::{Cell, RefCell};
use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet, VecDeque};
use std::ffi::CString;
use std::hash::{BuildHasherDefault, Hash, Hasher};
use std::io::{Read, Seek, Write};
#[cfg(unix)]
use std::os::unix::fs::{FileExt, MetadataExt, OpenOptionsExt};
use std::path::{Path, PathBuf};
use std::process::{Child, ChildStdout, Command, Stdio};
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Condvar, Mutex, OnceLock, RwLock};
use std::thread::ThreadId;
use std::time::{Duration, Instant};

use remu_core::GuestAddress;
use remu_loader::{
    DynamicInfo, Elf64Rela, Elf64Symbol, ImportedSymbol, InspectError, LoadedExecutable,
    ModuleDependencyGraphError, UnresolvedModuleDependency, resolve_module_dependency_graph,
};
#[cfg(target_os = "macos")]
use remu_memory::{AddressSpaceError, GUEST_PAGE_SIZE, GuestMapCoalescing, GuestMapPlacement};
use remu_memory::{GUEST_EXECUTE, GUEST_READ, GUEST_WRITE, GuestImage, MappingError};
use remu_shader::GuestMemoryReader;
use sha1::{Digest, Sha1};
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
const SHADER_EVALUATION_CACHE_CAPACITY: usize = 16 * 1024;
const SHADER_EVALUATION_PENDING_CAPACITY: usize = SHADER_EVALUATION_CACHE_CAPACITY * 4;

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
    bundled_modules: Vec<LinkedGuestModule>,
    unwind_modules: Vec<GuestUnwindModule>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct LinkedGuestModule {
    name: String,
    handle: u32,
    exports_by_nid: BTreeMap<String, u64>,
    initializers: Vec<GuestAddress>,
    started: bool,
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct GuestUnwindModule {
    name: String,
    base: u64,
    end: u64,
    eh_frame_header: u64,
    eh_frame: u64,
    eh_frame_size: u64,
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
    #[error("could not discover runtime modules under {path}: {source}")]
    RuntimeModuleDiscovery {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },
    #[error("guest {kind} byte size 0x{size:x} is not a multiple of eight")]
    InvalidInitializerArraySize { kind: &'static str, size: u64 },
    #[error("guest initializer address {address} is outside the process image")]
    InitializerOutsideImage { address: GuestAddress },
    #[error("native macOS C++ unwind runtime is unavailable")]
    NativeCxxRuntimeUnavailable,
    #[error("native macOS unwind symbol {0} is unavailable")]
    MissingNativeUnwindSymbol(&'static str),
}

fn discover_runtime_module_paths(app_root: &Path) -> Result<Vec<PathBuf>, LinkError> {
    let root = app_root.join("decrypted/Media/Plugins");
    if !root.is_dir() {
        return Ok(Vec::new());
    }
    let mut pending = vec![root];
    let mut modules = Vec::new();
    while let Some(directory) = pending.pop() {
        let entries = std::fs::read_dir(&directory)
            .map_err(|source| LinkError::RuntimeModuleDiscovery {
                path: directory.clone(),
                source,
            })?
            .collect::<Result<Vec<_>, _>>()
            .map_err(|source| LinkError::RuntimeModuleDiscovery {
                path: directory.clone(),
                source,
            })?;
        let mut entries = entries;
        entries.sort_by_key(std::fs::DirEntry::file_name);
        for entry in entries.into_iter().rev() {
            let path = entry.path();
            let file_type =
                entry
                    .file_type()
                    .map_err(|source| LinkError::RuntimeModuleDiscovery {
                        path: path.clone(),
                        source,
                    })?;
            if file_type.is_dir() {
                pending.push(path);
                continue;
            }
            if (!file_type.is_file() && !file_type.is_symlink())
                || !path
                    .extension()
                    .and_then(|extension| extension.to_str())
                    .is_some_and(|extension| {
                        extension.eq_ignore_ascii_case("prx")
                            || extension.eq_ignore_ascii_case("sprx")
                    })
            {
                continue;
            }
            modules.push(std::fs::canonicalize(&path).map_err(|source| {
                LinkError::RuntimeModuleDiscovery {
                    path: path.clone(),
                    source,
                }
            })?);
        }
    }
    modules.sort();
    Ok(modules)
}

impl LinkedImage {
    /// Discovers, loads, maps, and relocates the complete bundled dependency graph for `path`.
    ///
    /// Missing system PRXs remain explicit in the returned report because those dependencies are
    /// fulfilled by `REMu`'s HLE services rather than files from the game dump.
    pub fn link_path(path: impl AsRef<Path>) -> Result<(Self, ModuleLinkReport), LinkError> {
        let graph = resolve_module_dependency_graph(path)?;
        let executable = remu_loader::load_path(&graph.entry.path)?;
        let mut dependency_order = graph.dependency_order;
        let mut unresolved_dependencies = graph.unresolved_dependencies;
        let mut loaded_paths = dependency_order
            .iter()
            .map(|node| node.path.clone())
            .collect::<BTreeSet<_>>();
        let startup_paths = loaded_paths.clone();
        for plugin_path in discover_runtime_module_paths(&graph.app_root)? {
            let plugin_graph = resolve_module_dependency_graph(&plugin_path)?;
            unresolved_dependencies.extend(plugin_graph.unresolved_dependencies);
            for node in plugin_graph
                .dependency_order
                .into_iter()
                .chain(std::iter::once(plugin_graph.entry))
            {
                if loaded_paths.insert(node.path.clone()) {
                    dependency_order.push(node);
                }
            }
        }
        let mut modules = Vec::with_capacity(dependency_order.len());
        for node in &dependency_order {
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
        let module_startup = dependency_order
            .iter()
            .map(|node| startup_paths.contains(&node.path))
            .collect::<Vec<_>>();
        let image =
            Self::link_main_with_modules_and_startup(&executable, &module_refs, &module_startup)?;
        let report = ModuleLinkReport {
            app_root: graph.app_root,
            entry_path: graph.entry.path,
            loaded_modules: dependency_order.into_iter().map(|node| node.path).collect(),
            unresolved_dependencies,
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
        let module_startup = vec![true; modules.len()];
        Self::link_main_with_modules_and_startup(executable, modules, &module_startup)
    }

    fn link_main_with_modules_and_startup(
        executable: &LoadedExecutable,
        modules: &[(&str, &LoadedExecutable)],
        module_startup: &[bool],
    ) -> Result<Self, LinkError> {
        debug_assert_eq!(modules.len(), module_startup.len());
        FMOD_CORE_SYSTEM_INITIALIZE_TARGET.store(0, Ordering::Release);
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
        let mut linked_module_initializers = Vec::with_capacity(linked_modules.len());
        for module in &linked_modules {
            let mut initializers = Vec::new();
            append_object_initializers(&mut initializers, &memory, module.base, module.dynamic)?;
            linked_module_initializers.push(initializers);
        }
        let module_initializers = linked_module_initializers
            .iter()
            .zip(module_startup)
            .filter(|(_, startup)| **startup)
            .flat_map(|(initializers, _)| initializers.iter().copied())
            .collect();
        let bundled_modules = modules
            .iter()
            .zip(&linked_modules)
            .zip(&linked_module_initializers)
            .zip(module_startup)
            .map(|((((name, _), module), initializers), startup)| {
                Ok(LinkedGuestModule {
                    name: (*name).to_owned(),
                    handle: u32::try_from(module.tls_module_id)
                        .expect("the module count was already represented by a platform usize"),
                    exports_by_nid: module_exports_by_nid(&memory, module)?,
                    initializers: initializers.clone(),
                    started: *startup,
                })
            })
            .collect::<Result<Vec<_>, LinkError>>()?;
        let unwind_modules = std::iter::once(("eboot.bin", executable, main_base))
            .chain(
                modules
                    .iter()
                    .zip(&linked_modules)
                    .map(|((name, executable), module)| (*name, *executable, module.base)),
            )
            .map(|(name, executable, base)| guest_unwind_module(&memory, name, base, executable))
            .collect::<Result<Vec<_>, _>>()?;
        // A Gen5 main executable enters through its own runtime bootstrap. Its DT_INIT commonly
        // aliases the ELF entry point, so invoking it here would start the title once with null
        // process arguments and then start it a second time below. Dependency modules have no
        // independent process entry and therefore still require loader-driven initialization.
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
            bundled_modules,
            unwind_modules,
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

fn module_exports_by_nid(
    memory: &GuestImage,
    module: &LinkedModuleInput<'_>,
) -> Result<BTreeMap<String, u64>, LinkError> {
    let mut exports = BTreeMap::<String, ProcessExport>::new();
    for symbol in module.dynamic.symbols.iter().filter(|symbol| {
        !symbol.is_undefined() && !symbol.name.is_empty() && matches!(symbol.binding(), 1 | 2)
    }) {
        let nid = symbol.name.split('#').next().unwrap_or_default();
        if nid.is_empty() {
            continue;
        }
        let candidate = ProcessExport {
            address: defined_symbol_address_at(memory, module.base, symbol)?,
            weak: symbol.is_weak(),
        };
        match exports.get_mut(nid) {
            None => {
                exports.insert(nid.to_owned(), candidate);
            }
            Some(current) if current.weak && !candidate.weak => *current = candidate,
            Some(_) => {}
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

    use super::{ProcessSymbolKey, encode_sce_id, process_symbol_key, sce_symbol_nid};

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
    fn hashes_runtime_symbol_names_to_sony_nids() {
        assert_eq!(sce_symbol_nid("sceKernelOpen"), "1G3lF1Gg1k8");
        assert_eq!(sce_symbol_nid("sceKernelClose"), "UK2Tl2DWUns");
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

fn tls_dtv_entry_offset(module_id: u64) -> Option<u64> {
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
            if nid == "+IB8LyiKvIE"
                && std::env::var_os("REMU_TRACE_FMOD").is_some_and(|value| value == "1")
            {
                FMOD_CORE_SYSTEM_INITIALIZE_TARGET.store(
                    usize::try_from(address)
                        .expect("guest function address fits the native pointer width"),
                    Ordering::Release,
                );
                return Ok(trace_fmod_core_system_initialize as *const () as usize as u64);
            }
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

fn guest_unwind_module(
    memory: &GuestImage,
    name: &str,
    image_base: GuestAddress,
    executable: &LoadedExecutable,
) -> Result<GuestUnwindModule, LinkError> {
    let mut base = u64::MAX;
    let mut end = 0_u64;
    for segment in &executable.load_segments {
        if segment.memory_size == 0 {
            continue;
        }
        let start = runtime_address_at(
            memory,
            image_base,
            segment.virtual_address,
            usize::try_from(segment.memory_size).map_err(|_| LinkError::AddressOverflow)?,
        )?
        .get();
        base = base.min(start);
        end = end.max(
            start
                .checked_add(segment.memory_size)
                .ok_or(LinkError::AddressOverflow)?,
        );
    }
    if base == u64::MAX || end <= base {
        return Err(LinkError::Mapping(MappingError::NoLoadSegments));
    }

    let mut module = GuestUnwindModule {
        name: name.to_owned(),
        base,
        end,
        eh_frame_header: 0,
        eh_frame: 0,
        eh_frame_size: 0,
    };
    let Some(header) = executable.info.elf.as_ref().and_then(|elf| {
        elf.program_headers
            .iter()
            .find(|header| header.program_type == PT_GNU_EH_FRAME && header.file_size >= 8)
    }) else {
        return Ok(module);
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
    if frame_address < base || frame_address >= end {
        return Err(LinkError::MalformedEhFrame(
            ".eh_frame pointer is outside its module",
        ));
    }
    module.eh_frame_header = header_address.get();
    module.eh_frame = frame_address;
    module.eh_frame_size = if header_address.get() > frame_address {
        header_address.get() - frame_address
    } else {
        end - frame_address
    };
    Ok(module)
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
        "wzvqT4UqKX8" => "sceKernelLoadStartModule",
        "LwG8g3niqwA" => "sceKernelDlsym",
        "WkwEd3N7w0Y" => "sceKernelInstallExceptionHandler",
        "Qhv5ARAoOEc" => "sceKernelRemoveExceptionHandler",
        "il03nluKfMk" => "sceKernelRaiseException",
        "6xVpy0Fdq+I" => "_sigprocmask",
        "4J2sUJmuHZQ" => "sceKernelGetProcessTime",
        "fgxnMeTNUtY" => "sceKernelGetProcessTimeCounter",
        "BNowx2l588E" => "sceKernelGetProcessTimeCounterFrequency",
        "bnZxYgAFeA0" => "sceKernelGetSanitizerNewReplaceExternal",
        "vNe1w4diLCs" => "__tls_get_addr",
        "NWtTN10cJzE" => "sceLibcHeapGetTraceInfo",
        "-hn1tcVHq5Q" => "sceLibcMspaceCreate",
        "LYo3GhIlB38" => "sceLibcMspaceCalloc",
        "OJjm-QOIHlI" => "sceLibcMspaceMalloc",
        "Vla-Z+eXlxo" => "sceLibcMspaceFree",
        "W6SiVSiCDtI" => "sceLibcMspaceDestroy",
        "fEoW6BJsPt4" => "sceLibcMspaceMallocUsableSize",
        "gigoVHZvVPE" => "sceLibcMspaceRealloc",
        "iF1iQHzxBJU" => "sceLibcMspaceMemalign",
        "ljkqMcC4-mk" => "sceLibcMspaceAlignedAlloc",
        "p6lrRW8-MLY" => "sceLibcMspaceReallocalign",
        "pzUa7KEoydw" => "sceLibcMspaceIsHeapEmpty",
        "qWESlyXMI3E" => "sceLibcMspacePosixMemalign",
        "mkx2fVhNMsg" => "pthread_cond_broadcast",
        "kDh-NfxgMtE" => "scePthreadCondSignal",
        "JGgj7Uvrl+A" => "scePthreadCondBroadcast",
        "2MOy+rUfuhQ" => "pthread_cond_signal",
        "WKAXJ4XBPQ4" => "scePthreadCondWait",
        "Op8TBGY5KHg" => "pthread_cond_wait",
        "0TyVk4MSLt0" => "pthread_cond_init",
        "g+PZd2hiacg" => "scePthreadCondDestroy",
        "RXXqi4CtF8w" => "pthread_cond_destroy",
        "QxqK-IdpumU" => "_Getpmbstate",
        "zS94yyJRSUs" => "_Getpwcstate",
        "-9SIhUr4Iuo" => "_Mbtowcx",
        "stv1S3BKfgw" => "_Wctombx",
        "ATpTPg64JhM" => "_Wctombx.initial",
        "FxVZqBAA7ks" => "_write",
        "4wSze92BhLI" => "sceKernelWrite",
        "aI+OeCz8xrQ" => "scePthreadSelf",
        "EotR8a3ASf4" => "pthread_self",
        "3kg7rT0NQIs" => "scePthreadExit",
        "FJrT5LuUBAU" => "pthread_exit",
        "h9CcP3J0oVM" => "pthread_join",
        "7Xl257M4VNI" => "pthread_equal",
        "pO96TwzOm5E" => "sceKernelGetDirectMemorySize",
        "rTXw65xmLIA" => "sceKernelAllocateDirectMemory",
        "B+vc2AO2Zrc" => "sceKernelAllocateMainDirectMemory",
        "BHouLQzh0X0" => "sceKernelDirectMemoryQuery",
        "L-Q3LEjIbgA" => "sceKernelMapDirectMemory",
        "NcaWUxfMNIQ" => "sceKernelMapNamedDirectMemory",
        "mL8NDH86iQI" => "sceKernelMapNamedFlexibleMemory",
        "gEpBkcwxUjw" => "sceKernelAprResolveFilepathsToIdsAndFileSizes",
        "ASoW5WE-UPo" => "sceKernelAprSubmitCommandBufferAndGetResult",
        "rqwFKI4PAiM" => "sceKernelAprWaitCommandBuffer",
        "BQQniolj9tQ" => "sceKernelMapDirectMemory2",
        "hwVSPCmp5tM" => "sceKernelCheckedReleaseDirectMemory",
        "MBuItvba6z8" => "sceKernelReleaseDirectMemory",
        "7oxv3PPCumo" => "sceKernelReserveVirtualRange",
        "DGMG3JshrZU" => "sceKernelSetVirtualRangeName",
        "rVjRvHJ0X6c" => "sceKernelVirtualQuery",
        "cQke9UuBQOk" => "sceKernelMunmap",
        "2SKEx6bSq-4" => "sceKernelBatchMap",
        "kBJzF8x4SyE" => "sceKernelBatchMap2",
        "9bfdLIyuwCY" => "sceKernelMtypeprotect",
        "D0OdFMjp46I" => "sceKernelCreateEqueue",
        "fzyMKs9kim0" => "sceKernelWaitEqueue",
        "WDszmSbWuDk" => "sceKernelAddUserEventEdge",
        "bBfz7kMF2Ho" => "sceKernelAddAmprEvent",
        "kwGyyjohI50" => "sceKernelGetEventData",
        "Xs9hdiD7sAA" => "pthread_setschedparam",
        "w2rJhmD+dsE" => "sceAgcDriverAddEqEvent",
        "BpFoboUJoZU" => "sceKernelCreateEventFlag",
        "8mql9OcQnd4" => "sceKernelDeleteEventFlag",
        "IOnSvHzqu6A" => "sceKernelSetEventFlag",
        "7uhBFWRAS60" => "sceKernelClearEventFlag",
        "9lvj5DjHZiA" => "sceKernelPollEventFlag",
        "JTvBflhYazQ" => "sceKernelWaitEventFlag",
        "PZku4ZrXJqg" => "sceKernelCancelEventFlag",
        "lLMT9vJAck0" => "clock_gettime",
        "QBi7HCK03hw" => "sceKernelClockGettime",
        "ejekcaNQNq0" => "sceKernelGettimeofday",
        "0NTHN1NKONI" => "sceKernelConvertLocaltimeToUtc",
        "-o5uEDpN+oY" => "sceKernelConvertUtcToLocaltime",
        "n88vx3C5nW8" => "gettimeofday",
        "wLlFkwG9UcQ" => "time",
        "9BcDykPmo1I" => "__error",
        "HQOwnfMGipQ" => "sceNetErrnoLoc",
        "Q4qBuN-c0ZM" => "sceNetSocket",
        "dgJBaeJnGpo" => "sceNetPoolCreate",
        "hdpVEUDFW3s" => "sceSslInit",
        "3JCe3lCbQ8A" => "sceHttpInit",
        "+o9816YQhqQ" => "sceNpWebApi2Initialize",
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
        "1FGvU0i9saQ" => "scePthreadMutexattrSetprotocol",
        "dQHWEsJtoE4" => "pthread_mutexattr_init",
        "mDmgMOGVUqg" => "pthread_mutexattr_settype",
        "HF7lK46xzjY" => "pthread_mutexattr_destroy",
        "ytQULN-nhL4" => "pthread_rwlock_init",
        "1471ajPzxh0" => "pthread_rwlock_destroy",
        "iGjsr1WAtI0" => "pthread_rwlock_rdlock",
        "sIlRvQqsN2Y" => "pthread_rwlock_wrlock",
        "SFxTMOfuCkE" => "pthread_rwlock_tryrdlock",
        "XhWHn6P5R7U" => "pthread_rwlock_trywrlock",
        "EgmLo6EWgso" => "pthread_rwlock_unlock",
        "RpQJJVKTiFM" => "sceKernelGetModuleInfoForUnwind",
        "4fU5yvOkVG4" => "sceSysmoduleGetModuleInfoForUnwind",
        "crb5j7mkk1c" => "_is_signal_return",
        "OMDRKKAZ8I4" => "sceKernelDebugRaiseException",
        "mqULNdimTn0" => "pthread_key_create",
        "WrOLvHU0yQM" => "pthread_setspecific",
        "0-KXaS70xy4" => "pthread_getspecific",
        "PrdHuuDekhY" => "scePthreadKeyDelete",
        "6BpEZuDT7YI" => "pthread_key_delete",
        "bt3CTBKmGyI" => "scePthreadSetaffinity",
        "3qxgM4ezETA" => "scePthreadAttrSetaffinity",
        "W0Hpm2X0uPE" => "scePthreadSetprio",
        "cmo1RIYva9o" => "scePthreadMutexInit",
        "9UK1vLZQft4" => "scePthreadMutexLock",
        "tn3VlD0hG60" => "scePthreadMutexUnlock",
        "2Of0f+3mhhE" => "scePthreadMutexDestroy",
        "7H0iTOciTLo" => "pthread_mutex_lock",
        "K-jXhbt2gn4" => "pthread_mutex_trylock",
        "upoVrzMHFeE" => "scePthreadMutexTrylock",
        "2Z+PpY6CaJg" => "pthread_mutex_unlock",
        "nsYoNRywwNg" => "scePthreadAttrInit",
        "wtkt-teR1so" => "pthread_attr_init",
        "62KCwEMmzcM" => "scePthreadAttrDestroy",
        "zHchY8ft5pk" => "pthread_attr_destroy",
        "-Wreprtu0Qs" => "scePthreadAttrSetdetachstate",
        "UTXzJbWhhTE" => "scePthreadAttrSetstacksize",
        "2Q0z6rnBrTE" => "pthread_attr_setstacksize",
        "DzES9hQF4f4" => "scePthreadAttrSetschedparam",
        "FXPWHNk8Of0" => "scePthreadAttrGetschedparam",
        "eXbUSpEaTsA" => "scePthreadAttrSetinheritsched",
        "4+h9EzwKF4I" => "scePthreadAttrSetschedpolicy",
        "P41kTWUS3EI" => "scePthreadGetschedparam",
        "oIRFTjoILbg" => "scePthreadSetschedparam",
        "GBUY7ywdULE" => "scePthreadRename",
        "x1X76arYMxU" => "scePthreadAttrGet",
        "-quPa4SEJUw" => "scePthreadAttrGetstack",
        "Ru36fiTtJzA" => "scePthreadAttrGetstackaddr",
        "-fA+7ZlGDQs" => "scePthreadAttrGetstacksize",
        "8+s5BzZjxSg" => "scePthreadAttrGetaffinity",
        "m5-2bsNfv7s" => "scePthreadCondattrInit",
        "waPcxYiR3WA" => "scePthreadCondattrDestroy",
        "2Tb92quprl0" => "scePthreadCondInit",
        "BmMjYxmew1w" => "scePthreadCondTimedwait",
        "4qGrR6eoP9Y" => "scePthreadDetach",
        "+U1R4WtXvoc" => "pthread_detach",
        "T72hz6ffq08" => "scePthreadYield",
        "14bOACANTBo" => "scePthreadOnce",
        "6UgtwV+0zb4" => "scePthreadCreate",
        "OxhIB8LB-PQ" => "pthread_create",
        "onNY9Byn-W8" => "scePthreadJoin",
        "188x57JYp0g" => "sceKernelCreateSema",
        "R1Jvn8bSCW8" => "sceKernelDeleteSema",
        "Xjoosiw+XPI" => "sceKernelUuidCreate",
        "yDBwVAolDgg" => "sceKernelIsStack",
        "1-LFLmRFxxM" => "sceKernelMkdir",
        "Qs0wWulgl7U" => "sceMouseInit",
        "RaqxZIf6DvE" => "sceMouseOpen",
        "rbknaUjpqWo" => "sceNpGetAccountIdA",
        "JT+t00a3TxA" => "sceNpGetAccountCountryA",
        "eQH7nWPcAgc" => "sceNpGetState",
        "e-ZuhGEoeC4" => "sceNpGetNpReachabilityState",
        "1xxcMiGu2fo" => "sceUserServiceGetUserName",
        "woNpu+45RLk" => "sceUserServiceGetAgeLevel",
        "-sD02mFDBh4" => "sceUserServiceGetGamePresets",
        "yH17Q6NWtVg" => "sceUserServiceGetEvent",
        "R9lA82OraNs" => "sceAppContentInitialize",
        "99b82IKXpH4" => "sceAppContentAppParamGetInt",
        "eaFXjfJv3xs" => "sceImeKeyboardOpen",
        "-4GCfYdNF1s" => "sceImeUpdate",
        "18B2NS1y9UU" => "sceRtcGetCurrentTick",
        "ZPD1YOKI+Kw" => "sceRtcGetCurrentClockLocalTime",
        "8w-H19ip48I" => "sceRtcGetTick",
        "ueega6v3GUw" => "sceRtcSetTick",
        "1n37q1Bvc5Y" => "sceSystemServiceGetDisplaySafeAreaInfo",
        "naInUjYt3so" => "sceKernelRmdir",
        "27bAgiJmOh0" => "pthread_cond_timedwait",
        "9vA2aW+CHuA" => "sceNetInetNtop",
        "C4UgDHHPvdw" => "sceNetResolverCreate",
        "obuxdTiwkF8" => "sceNetCtlGetInfo",
        "A9cVMUtEp4Y" => "sceHttpInit",
        "0gYjPTR-6cY" => "sceHttpCreateTemplate",
        "nBDD66kiFW8" => "sceShareInitialize",
        "8zLSfEfW5AU" => "sceCoredumpRegisterCoredumpHandler",
        "j2AIqSqJP0w" => "sceKernelGetdents",
        "2G6i6hMIUUY" => "posix_getdents",
        "taRWhTJFTgE" => "sceKernelGetdirentries",
        "sfKygSjIbI8" => "getdirentries",
        "ts6GlZOKRrE" => "scePlayGoInitialize",
        "MPe0EeBGM-E" => "scePlayGoTerminate",
        "M1Gma1ocrGE" => "scePlayGoOpen",
        "Uco1I0dlDi8" => "scePlayGoClose",
        "uWIYLFkkwqk" => "scePlayGoGetLocus",
        "-RJWNMK3fC8" => "scePlayGoGetProgress",
        "v6EZ-YWRdMs" => "scePlayGoGetEta",
        "rvBSfTimejE" => "scePlayGoGetInstallSpeed",
        "4AAcTU9R3XM" => "scePlayGoSetInstallSpeed",
        "73fF1MFU8hA" => "scePlayGoGetChunkId",
        "3OMbYZBaa50" => "scePlayGoGetLanguageMask",
        "LosLlHOpNqQ" => "scePlayGoSetLanguageMask",
        "Nn7zKwnA5q0" => "scePlayGoGetToDoList",
        "gUPGiOQ1tmQ" => "scePlayGoSetToDoList",
        "-Q1-u1a7p0g" => "scePlayGoPrefetch",
        "lDqxaY1UbEo" => "sceMsgDialogInitialize",
        "ePw-kqZmelo" => "sceMsgDialogTerminate",
        "b06Hh0DPEaE" => "sceMsgDialogOpen",
        "HTrcDKlFKuM" => "sceMsgDialogClose",
        "CWVW78Qc3fI" => "sceMsgDialogGetStatus",
        "6fIC3XKt2k0" => "sceMsgDialogUpdateStatus",
        "Lr8ovHH9l6A" => "sceMsgDialogGetResult",
        "nu4a0-arQis" => "sceKernelAioInitializeParam",
        "vYU8P9Td2Zo" => "sceKernelAioInitializeImpl",
        "geDaqgH9lTg" => "posix_pthread_key_create",
        "EI-5-jlq2dE" => "posix_pthread_getthreadid_np",
        "eR2bZFAAU0Q" => "sceSysmoduleUnloadModule",
        "buYbeLOGWmA" => "sceAppContentTemporaryDataMount2",
        "zE-wXIZjLoM" => "sceKernelDebugRaiseExceptionOnReleaseMode",
        "dKadqZFgKKQ" => "sceImeKeyboardGetResourceId",
        "VkqLPArfFdc" => "sceImeKeyboardGetInfo",
        "rnEhHqG-4xo" => "sceUserServiceGetAccessibilityChatTranscription",
        "ZKJtxdgvzwg" => "sceUserServiceGetAccessibilityPressAndHoldDelay",
        "-3Y5GO+-i78" => "sceUserServiceGetAccessibilityTriggerEffect",
        "qWYHOFwqCxY" => "sceUserServiceGetAccessibilityVibration",
        "hD-H81EN9Vg" => "sceUserServiceGetAccessibilityZoomEnabled",
        "O6IW1-Dwm-w" => "sceUserServiceGetAccessibilityZoomFollowFocus",
        "Zxa0VhQVTsk" => "sceKernelWaitSema",
        "4czppHBiriw" => "sceKernelSignalSema",
        "1jfXLRVzisc" => "sceKernelUsleep",
        "g8cM39EUZ6o" => "sceSysmoduleLoadModule",
        "fMP5NHUOaMk" => "sceSysmoduleIsLoaded",
        "YJtKLttI9fM" => "sceGameUpdateInitialize",
        "jO8DM8oyego" => "sceNpEntitlementAccessInitialize",
        "ysmw6J-P8Ak" => "sceNpSessionSignalingInitialize",
        "sk54bi6FtYM" => "sceNpWebApi2CreateUserContext",
        "3Zl8BePTh9Y" => "sceNpCheckCallback",
        "s6W4Zl4Slgk" => "sceNpUniversalDataSystemCreateEventPropertyObject",
        "hZIg1EWGsHM" => "sceUltInitialize",
        "CdWp0oHWGr0" => "sceUserServiceGetInitialUser",
        "fPhymKNvK-A" => "sceUserServiceGetLoginUserIdList",
        "j3YMu1MVNNo" => "sceUserServiceInitialize",
        "hv1luiJrqQM" => "scePadInit",
        "xk0AcarP3V4" => "scePadOpen",
        "6ncge5+l5Qs" => "scePadClose",
        "AcslpN1jHR8" => "scePadDeviceClassGetExtendedInformation",
        "clVvL4ZDntw" => "scePadSetMotionSensorState",
        "gjP9-KQzoUk" => "scePadGetControllerInformation",
        "YndgXqQVV7c" => "scePadReadState",
        "q1cHNfGycLI" => "scePadRead",
        "W2G-yoyMF5U" => "scePadSetVibrationMode",
        "rIZnR6eSpvk" => "scePadResetOrientation",
        "r44mAxdSG+U" => "scePadSetAngularVelocityDeadbandState",
        "vDLMoJLde8I" => "scePadSetTiltCorrectionState",
        "yFVnOdGxvZY" => "scePadSetVibration",
        "2JgFB2n9oUM" => "scePadSetTriggerEffect",
        "znaWI0gpuo8" => "scePadGetTriggerEffectState",
        "RR4novUEENY" => "scePadSetLightBar",
        "DscD1i9HX1w" => "scePadResetLightBar",
        "IHPqcbc0zCA" => "scePadDeviceClassParseData",
        "x8qnXqh-tiM" => "sceMouseRead",
        "JfEPXVxhFqA" => "sceAudioOutInit",
        "ekNvsT22rsY" => "sceAudioOutOpen",
        "s1--uE9mBFw" => "sceAudioOutClose",
        "b+uAV89IlxE" => "sceAudioOutSetVolume",
        "QOQtbeDqsT4" => "sceAudioOutOutput",
        "GrQ9s4IrNaQ" => "sceAudioOutGetPortState",
        "KMcEa+rHsIo" => "sceAvPlayerAddSource",
        "x8uvuFOPZhU" => "sceAvPlayerAddSourceEx",
        "NkJwDzKmIlw" => "sceAvPlayerClose",
        "wwM99gjFf1Y" => "sceAvPlayerCurrentTime",
        "Wnp1OVcrZgk" => "sceAvPlayerGetAudioData",
        "d8FcbzfAdQw" => "sceAvPlayerGetStreamInfo",
        "o3+RWnHViSg" => "sceAvPlayerGetVideoData",
        "JdksQu8pNdQ" => "sceAvPlayerGetVideoDataEx",
        "ctTAcF5DiKQ" => "sceAvPlayerGetStreamInfoEx",
        "ODJK2sn9w4A" => "sceAvPlayerEnableStream",
        "k-q+xOxdc3E" => "sceAvPlayerSetAvSyncMode",
        "aS66RI0gGgo" => "sceAvPlayerInit",
        "o9eWRkSL+M4" => "sceAvPlayerInitEx",
        "HD1YKVU26-M" => "sceAvPlayerPostInit",
        "UbQoYawOsfY" => "sceAvPlayerIsActive",
        "XC9wM+xULz8" => "sceAvPlayerJumpToTime",
        "9y5v+fGN4Wk" => "sceAvPlayerPause",
        "w5moABNwnRY" => "sceAvPlayerResume",
        "eBTreZ84JFY" => "sceAvPlayerSetLogCallback",
        "OVths0xGfho" => "sceAvPlayerSetLooping",
        "ET4Gr-Uu07s" => "sceAvPlayerStart",
        "ZC17w3vB5Lo" => "sceAvPlayerStop",
        "hdTyRzCXQeQ" => "sceAvPlayerStreamCount",
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
        "0x6o1VVAYSY" => "sceAudioOut2ContextCreate",
        "8XTArSPyWHk" => "sceAudioOut2PortSetAttributes",
        "IaZXJ9M79uo" => "sceAudioOut2UserDestroy",
        "JK2wamZPzwM" => "sceAudioOut2PortCreate",
        "PE2zHMqLSHs" => "sceAudioOut2ContextAdvance",
        "aII9h5nli9U" => "sceAudioOut2ContextPush",
        "cd+Rtw+D1x8" => "sceAudioOut2PortDestroy",
        "g2tViFIohHE" => "sceAudioOut2Initialize",
        "on6ZH7Abo10" => "sceAudioOut2ContextDestroy",
        "pDmme7Bgm6E" => "sceAudioOut2ContextQueryMemory",
        "t5YrizufpQc" => "sceAudioOut2ContextResetParam",
        "xywYcRB7nbQ" => "sceAudioOut2UserCreate",
        "-qLsfDAywIY" => "sceAjmBatchWait",
        "39WxhR-ePew" => "sceAjmBatchJobDecode",
        "5tOfnaClcqM" => "sceAjmBatchStart",
        "AxoDrINp4J8" => "sceAjmInstanceCreate",
        "MHur6qCsUus" => "sceAjmFinalize",
        "MmpF1XsQiHw" => "sceAjmBatchInitialize",
        "Q3dyFuwGn64" => "sceAjmModuleRegister",
        "RbLbuKv8zho" => "sceAjmInstanceDestroy",
        "SkEwpiu3tZg" => "sceAjmBatchJobSetGaplessDecode",
        "WfAiBW8Wcek" => "sceAjmBatchErrorDump",
        "Wi7DtlLV+KI" => "sceAjmModuleUnregister",
        "dl+4eHSzUu4" => "sceAjmInitialize",
        "ezM2OhNxzck" => "sceAjmBatchJobInitialize",
        "cK6bYHf-Q5E" => "sce::Json::Initializer::Initializer",
        "-hJRce8wn1U" => "sce::Json::MemAllocator::MemAllocator",
        "WSOuge5IsCg" => "sce::Json::InitParameter2::InitParameter2",
        "I2QC8PYhJWY" => "sce::Json::InitParameter2::setAllocator",
        "Eu95jmqn5Rw" => "sce::Json::InitParameter2::setFileBufferSize",
        "IXW-z8pggfg" => "sce::Json::Initializer::initialize(InitParameter2)",
        "Cxwy7wHq4J0" => "sce::Json::Initializer::initialize(InitParameter)",
        "PR5k1penBLM" => "sce::Json::Initializer::terminate",
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
        "w6Dj1VJt5qY" => "sceAgcSetPacketPredication",
        "n8vgpaQg6dA" => "sceAgcSetRangePredication",
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
        "utPrVdxio-8" => "sceVideoOutGetOutputStatus",
        "CBiu4mCE1DA" => "sceVideoOutSetFlipRate",
        "U46NwOiJpys" => "sceVideoOutSubmitFlip",
        "zgXifHT9ErY" => "sceVideoOutIsFlipPending",
        "8aI7R7WaOlc" => "sceAmprCommandBufferConstructor",
        "a8uLzYY--tM" => "sceAmprAprCommandBufferConstructor",
        "N-FSPA4S3nI" => "sceAmprCommandBufferSetBuffer",
        "baQO9ez2gL4" => "sceAmprCommandBufferReset",
        "ULvXMDz56po" => "sceAmprCommandBufferClearBuffer",
        "tZDDEo2tE5k" => "sceAmprCommandBufferGetSize",
        "GnxKOHEawhk" => "sceAmprCommandBufferGetCurrentOffset",
        "H896Pt-yB4I" => "sceAmprCommandBufferWriteKernelEventQueue_04_00",
        "sSAUCCU1dv4" => "sceAmprMeasureCommandSizeWriteKernelEventQueue_04_00",
        "vWU-odnS+fU" => "sceAmprMeasureCommandSizeReadFile",
        "GuchCTefuZw" => "sceAmprCommandBufferDestructor",
        "Qs1xtplKo0U" => "sceAmprAprCommandBufferDestructor",
        "Jmi+9w9u0E4" => "pthread_create_name_np",
        "mQ16-QdKv7k" => "sceAmprAprCommandBufferReadFile",
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
    #[cfg(target_os = "macos")]
    process_mapping_start: u64,
    #[cfg(target_os = "macos")]
    process_mapping_end: u64,
    next_thread_unique_id: u64,
    thread_handles: BTreeMap<u64, std::thread::JoinHandle<u64>>,
    thread_attributes: BTreeMap<u64, PthreadAttrState>,
    detached_threads: BTreeSet<u64>,
    pthread_threads: BTreeMap<u64, PthreadThreadObject>,
    pthread_attributes: BTreeMap<u64, PthreadAttrObject>,
    pthread_mutex_attributes: BTreeMap<u64, PthreadMutexAttrObject>,
    pthread_condition_attributes: BTreeSet<u64>,
    next_pthread_key: u32,
    pthread_keys: BTreeMap<u32, u64>,
    ampr_files: BTreeMap<u32, PathBuf>,
    ampr_file_handles: BTreeMap<u32, std::fs::File>,
    ampr_command_buffers: BTreeMap<u64, AmprCommandBufferState>,
    next_apr_submission_id: u32,
    apr_submissions: BTreeSet<u32>,
    exit_callbacks: Vec<GuestExitCallback>,
    /// Registered guest coredump handler and its requested stack size.
    ///
    /// The console runs this on a dedicated thread when the process faults. REmu records the
    /// registration so the ABI's already-registered and parameter errors are exact; generating a
    /// coredump to invoke it is a separate unimplemented capability.
    coredump_handler: Option<(u64, u64)>,
    process_param_address: u64,
    thread_atexit_count_callback: u64,
    thread_atexit_report_callback: u64,
    thread_dtors_callback: u64,
    application_heap_api_address: u64,
    bundled_modules: Vec<LinkedGuestModule>,
    unwind_modules: Vec<GuestUnwindModule>,
    installed_exception_handlers: BTreeMap<i32, u64>,
    next_ajm_context: u32,
    ajm_contexts: BTreeMap<u32, AjmContextState>,
    heap_trace_storage: Vec<u64>,
    guest_mspaces: BTreeMap<u64, GuestMspace>,
    json_runtime: JsonRuntimeState,
    av_player: AvPlayerRuntimeState,
    services: RuntimeServiceState,
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
    next_equeue: u64,
    kernel_equeues: BTreeSet<u64>,
    kernel_event_registrations: BTreeMap<(u64, i16, u64), u64>,
    pending_events: VecDeque<KernelQueuedEvent>,
    pending_agc_completions: VecDeque<u64>,
    save_root: PathBuf,
    save_data_initialized: bool,
    save_mounts: BTreeMap<String, SaveDataMount>,
    next_save_transaction_resource: u32,
    save_transaction_resources: BTreeSet<u32>,
    prepared_save_resources: BTreeSet<u32>,
    save_transactions: BTreeMap<u32, SaveDataTransaction>,
}

#[derive(Debug)]
struct GuestMspace {
    name: String,
    base: u64,
    end: u64,
    free_ranges: BTreeMap<u64, u64>,
    allocations: BTreeMap<u64, u64>,
}

#[derive(Debug, Default)]
struct JsonRuntimeState {
    allocators: BTreeMap<u64, (u64, u64)>,
    file_buffer_sizes: BTreeMap<u64, u64>,
    initialized_objects: BTreeMap<u64, u64>,
}

#[derive(Debug, Default)]
struct AvPlayerRuntimeState {
    log_callback: Option<(u64, u64)>,
    next_handle: u64,
    players: BTreeMap<u64, AvPlayerInstance>,
    pending_callbacks: Vec<AvPlayerEventCallback>,
}

#[derive(Clone, Copy, Debug)]
struct AvPlayerEventCallback {
    function: u64,
    object: u64,
    event: u64,
}

#[derive(Debug)]
struct AvPlayerInstance {
    _initialization: Vec<u8>,
    allocator_object: u64,
    allocate_texture_callback: u64,
    event_object: u64,
    event_callback: u64,
    source: Option<String>,
    auto_start: bool,
    width: u32,
    height: u32,
    frames_per_second: f64,
    duration: Duration,
    paused: bool,
    end_of_stream: bool,
    playing_since: Option<Instant>,
    elapsed: Duration,
    looping: bool,
    video_decoder: Option<Child>,
    video_output: Option<ChildStdout>,
    raw_video_frame: Vec<u8>,
    padded_video_frame: Vec<u8>,
    video_buffer_base: u64,
    video_buffers: [u64; 3],
    video_buffer_stride: usize,
    next_video_buffer: usize,
    next_video_frame: u64,
    audio_decoder: Option<Child>,
    audio_output: Option<ChildStdout>,
    raw_audio_frame: Vec<u8>,
    audio_buffer_base: u64,
    next_audio_buffer: usize,
    next_audio_frame: u64,
}

impl AvPlayerInstance {
    fn stop_decoders(&mut self) {
        self.video_output = None;
        self.audio_output = None;
        if let Some(mut decoder) = self.video_decoder.take() {
            let _ = decoder.kill();
            let _ = decoder.wait();
        }
        if let Some(mut decoder) = self.audio_decoder.take() {
            let _ = decoder.kill();
            let _ = decoder.wait();
        }
        self.playing_since = None;
    }

    fn reset_playback(&mut self) {
        self.stop_decoders();
        self.elapsed = Duration::ZERO;
        self.end_of_stream = false;
        self.next_video_frame = 0;
        self.next_audio_frame = 0;
        self.next_video_buffer = 0;
        self.next_audio_buffer = 0;
    }

    fn playback_elapsed(&self) -> Duration {
        self.playing_since
            .map_or(self.elapsed, |started| self.elapsed + started.elapsed())
    }
}

impl Drop for AvPlayerInstance {
    fn drop(&mut self) {
        self.stop_decoders();
        guest_free(self.video_buffer_base);
        guest_free(self.audio_buffer_base);
    }
}

impl GuestMspace {
    const ALLOCATION_ALIGNMENT: u64 = 16;
    const HEADER_SIZE: u64 = 0x100;

    fn new(name: String, base: u64, capacity: u64) -> Option<Self> {
        let end = base.checked_add(capacity)?;
        let first = align_up_u64(
            base.checked_add(Self::HEADER_SIZE)?,
            Self::ALLOCATION_ALIGNMENT,
        )?;
        if first >= end {
            return None;
        }
        let mut free_ranges = BTreeMap::new();
        free_ranges.insert(first, end - first);
        Some(Self {
            name,
            base,
            end,
            free_ranges,
            allocations: BTreeMap::new(),
        })
    }

    fn allocate(&mut self, requested_size: u64, alignment: u64) -> Option<u64> {
        if !alignment.is_power_of_two() {
            return None;
        }
        let alignment = alignment.max(Self::ALLOCATION_ALIGNMENT);
        let size = align_up_u64(requested_size.max(1), Self::ALLOCATION_ALIGNMENT)?;
        let candidate = self.free_ranges.iter().find_map(|(&start, &length)| {
            let aligned = align_up_u64(start, alignment)?;
            let allocation_end = aligned.checked_add(size)?;
            let range_end = start.checked_add(length)?;
            (allocation_end <= range_end).then_some((start, length, aligned, allocation_end))
        })?;
        let (range_start, range_length, allocation_start, allocation_end) = candidate;
        let range_end = range_start + range_length;
        self.free_ranges.remove(&range_start);
        if allocation_start > range_start {
            self.free_ranges
                .insert(range_start, allocation_start - range_start);
        }
        if allocation_end < range_end {
            self.free_ranges
                .insert(allocation_end, range_end - allocation_end);
        }
        self.allocations.insert(allocation_start, size);
        Some(allocation_start)
    }

    fn free(&mut self, address: u64) -> bool {
        let Some(size) = self.allocations.remove(&address) else {
            return address == 0;
        };
        self.insert_free_range(address, size);
        true
    }

    fn insert_free_range(&mut self, mut start: u64, mut size: u64) {
        if let Some((&previous_start, &previous_size)) = self.free_ranges.range(..start).next_back()
            && previous_start.checked_add(previous_size) == Some(start)
        {
            self.free_ranges.remove(&previous_start);
            start = previous_start;
            size += previous_size;
        }
        if let Some((&next_start, &next_size)) = self.free_ranges.range(start..).next()
            && start.checked_add(size) == Some(next_start)
        {
            self.free_ranges.remove(&next_start);
            size += next_size;
        }
        self.free_ranges.insert(start, size);
    }

    fn realloc(&mut self, address: u64, requested_size: u64) -> Option<u64> {
        if address == 0 {
            return self.allocate(requested_size, Self::ALLOCATION_ALIGNMENT);
        }
        if requested_size == 0 {
            self.free(address);
            return Some(0);
        }
        let old_size = *self.allocations.get(&address)?;
        let new_size = align_up_u64(requested_size, Self::ALLOCATION_ALIGNMENT)?;
        if new_size <= old_size {
            self.allocations.insert(address, new_size);
            if new_size < old_size {
                self.insert_free_range(address + new_size, old_size - new_size);
            }
            return Some(address);
        }
        let replacement = self.allocate(new_size, Self::ALLOCATION_ALIGNMENT)?;
        // SAFETY: Both ranges are disjoint, live allocations within the caller-provided mspace.
        unsafe {
            std::ptr::copy_nonoverlapping(
                address as *const u8,
                replacement as *mut u8,
                usize::try_from(old_size).ok()?,
            );
        }
        self.free(address);
        Some(replacement)
    }

    fn usable_size(&self, address: u64) -> u64 {
        self.allocations.get(&address).copied().unwrap_or(0)
    }
}

#[derive(Debug, Default)]
struct AjmContextState {
    registered_codecs: BTreeSet<u32>,
    next_instance: u32,
    instances: BTreeSet<u32>,
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

#[derive(Clone, Debug, Default, Eq, PartialEq)]
struct AmprCommandBufferState {
    buffer: u64,
    size: u64,
    write_offset: u64,
    command_count: u64,
    /// Queue completions recorded into this buffer, announced when it is submitted.
    pending_completions: Vec<AmprCompletion>,
}

/// One recorded request to signal an event queue when a command buffer completes.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct AmprCompletion {
    equeue: u64,
    ident: u64,
    user_data: u64,
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
    user_login_event_delivered: bool,
    pad: InitializationState,
    common_dialog: InitializationState,
    /// Lifecycle state of the system message dialog, using the common-dialog status values.
    message_dialog_status: u32,
    play_go_initialized: bool,
    play_go_open: bool,
    save_data_dialog_status: i32,
    agc_states: BTreeMap<u64, u32>,
    game_update: InitializationState,
    np_entitlement_access: Option<(u64, u64)>,
    np_session_signaling: Option<Vec<u8>>,
    next_np_web_api2_user_context: i32,
    np_web_api2_user_contexts: BTreeMap<i32, (i32, i32)>,
    np_callback_polls: u64,
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

#[derive(Clone, Debug, Eq, Hash, PartialEq)]
struct ShaderEvaluationCacheKey {
    program_address: u64,
    user_data: ShaderEvaluationUserDataKey,
    compute_system_registers: Option<remu_shader::ComputeSystemRegisters>,
    user_data_scalar_register_base: u32,
    options: remu_shader::EvaluationOptions,
}

#[derive(Clone, Debug)]
struct ShaderEvaluationUserDataKey {
    words: [u32; 64],
    len: u8,
}

impl ShaderEvaluationUserDataKey {
    fn as_slice(&self) -> &[u32] {
        &self.words[..usize::from(self.len)]
    }
}

impl PartialEq for ShaderEvaluationUserDataKey {
    fn eq(&self, other: &Self) -> bool {
        self.len == other.len && self.as_slice() == other.as_slice()
    }
}

impl Eq for ShaderEvaluationUserDataKey {}

impl Hash for ShaderEvaluationUserDataKey {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.len.hash(state);
        self.as_slice().hash(state);
    }
}

// Shader evaluation keys are entirely emulator-owned integers. SipHash's collision resistance is
// unnecessary here and is costly because the hot miss path hashes up to 64 user-data words in as
// many as three cache tables. This small Fx-style mixer retains full equality checks in HashMap
// while making those repeated fixed-width writes substantially cheaper.
#[derive(Debug)]
struct ShaderEvaluationCacheHasher(u64);

impl Default for ShaderEvaluationCacheHasher {
    fn default() -> Self {
        Self(0x517c_c1b7_2722_0a95)
    }
}

impl ShaderEvaluationCacheHasher {
    #[inline]
    fn mix(&mut self, value: u64) {
        self.0 = (self.0.rotate_left(5) ^ value).wrapping_mul(0x517c_c1b7_2722_0a95);
    }
}

impl Hasher for ShaderEvaluationCacheHasher {
    #[inline]
    fn finish(&self) -> u64 {
        self.0
    }

    fn write(&mut self, bytes: &[u8]) {
        let mut chunks = bytes.chunks_exact(8);
        for chunk in &mut chunks {
            let mut word = [0_u8; 8];
            word.copy_from_slice(chunk);
            self.mix(u64::from_ne_bytes(word));
        }
        let remainder = chunks.remainder();
        if !remainder.is_empty() {
            let mut word = [0_u8; 8];
            word[..remainder.len()].copy_from_slice(remainder);
            self.mix(u64::from_ne_bytes(word) ^ ((remainder.len() as u64) << 56));
        }
    }

    #[inline]
    fn write_u8(&mut self, value: u8) {
        self.mix(u64::from(value));
    }

    #[inline]
    fn write_u32(&mut self, value: u32) {
        self.mix(u64::from(value));
    }

    #[inline]
    fn write_u64(&mut self, value: u64) {
        self.mix(value);
    }

    #[inline]
    fn write_usize(&mut self, value: usize) {
        self.mix(value as u64);
    }
}

type ShaderEvaluationCacheBuildHasher = BuildHasherDefault<ShaderEvaluationCacheHasher>;

type ShaderEvaluationCacheValue = Result<Arc<remu_shader::ShaderEvaluation>, Arc<str>>;

fn retire_shader_evaluation(value: ShaderEvaluationCacheValue) {
    static RETIRER: OnceLock<std::sync::mpsc::Sender<ShaderEvaluationCacheValue>> = OnceLock::new();
    let sender = RETIRER.get_or_init(|| {
        let (sender, receiver) = std::sync::mpsc::channel::<ShaderEvaluationCacheValue>();
        if let Err(error) = std::thread::Builder::new()
            .name("REMu shader cache reclaimer".into())
            .spawn(move || {
                while let Ok(value) = receiver.recv() {
                    drop(value);
                }
            })
        {
            eprintln!("[REMu][AGC] could not start shader cache reclaimer: {error}");
        }
        sender
    });
    if let Err(error) = sender.send(value) {
        drop(error.0);
    }
}

#[derive(Debug)]
struct ShaderEvaluationCacheEntry {
    value: ShaderEvaluationCacheValue,
    admission_id: u64,
}

type ShaderEvaluationCacheEntries =
    HashMap<ShaderEvaluationCacheKey, ShaderEvaluationCacheEntry, ShaderEvaluationCacheBuildHasher>;
type ShaderEvaluationPendingEntries =
    HashMap<ShaderEvaluationCacheKey, u64, ShaderEvaluationCacheBuildHasher>;

#[derive(Debug, Default)]
struct ShaderEvaluationCache {
    entries: ShaderEvaluationCacheEntries,
    seen_once: ShaderEvaluationPendingEntries,
    pending_admission_order: VecDeque<(ShaderEvaluationCacheKey, u64)>,
    admission_order: VecDeque<(ShaderEvaluationCacheKey, u64)>,
    next_admission_id: u64,
    hits: u64,
    misses: u64,
    failures: u64,
    invalidations: u64,
    evictions: u64,
}

#[derive(Clone, Copy, Debug, Default)]
struct ShaderEvaluationCacheStats {
    entries: usize,
    pending_admissions: usize,
    hits: u64,
    misses: u64,
    failures: u64,
    invalidations: u64,
    evictions: u64,
}

impl ShaderEvaluationCache {
    fn stats(&self) -> ShaderEvaluationCacheStats {
        ShaderEvaluationCacheStats {
            entries: self.entries.len(),
            pending_admissions: self.seen_once.len(),
            hits: self.hits,
            misses: self.misses,
            failures: self.failures,
            invalidations: self.invalidations,
            evictions: self.evictions,
        }
    }
}

impl ShaderEvaluationCacheStats {
    fn merge(self, other: Self) -> Self {
        Self {
            entries: self.entries.saturating_add(other.entries),
            pending_admissions: self
                .pending_admissions
                .saturating_add(other.pending_admissions),
            hits: self.hits.saturating_add(other.hits),
            misses: self.misses.saturating_add(other.misses),
            failures: self.failures.saturating_add(other.failures),
            invalidations: self.invalidations.saturating_add(other.invalidations),
            evictions: self.evictions.saturating_add(other.evictions),
        }
    }
}

struct ShaderEvaluationWorkerRequest {
    state: remu_shader::ShaderState,
    options: remu_shader::EvaluationOptions,
}

struct ShaderEvaluationWorkerResult {
    evaluation: Result<Arc<remu_shader::ShaderEvaluation>, String>,
    stats: ShaderEvaluationCacheStats,
}

#[derive(Debug)]
struct ShaderEvaluationWorker {
    requests: std::sync::mpsc::SyncSender<ShaderEvaluationWorkerRequest>,
    results: Mutex<std::sync::mpsc::Receiver<ShaderEvaluationWorkerResult>>,
    stats: ShaderEvaluationCacheStats,
}

impl ShaderEvaluationWorker {
    fn spawn(name: &str) -> Result<Self, String> {
        let (requests, request_receiver) =
            std::sync::mpsc::sync_channel::<ShaderEvaluationWorkerRequest>(1);
        let (result_sender, results) =
            std::sync::mpsc::sync_channel::<ShaderEvaluationWorkerResult>(1);
        std::thread::Builder::new()
            .name(name.to_owned())
            .spawn(move || {
                let mut cache = ShaderEvaluationCache::default();
                while let Ok(request) = request_receiver.recv() {
                    let evaluation = cache.evaluate(&request.state, request.options);
                    if result_sender
                        .send(ShaderEvaluationWorkerResult {
                            evaluation,
                            stats: cache.stats(),
                        })
                        .is_err()
                    {
                        break;
                    }
                }
            })
            .map_err(|error| format!("could not start {name}: {error}"))?;
        Ok(Self {
            requests,
            results: Mutex::new(results),
            stats: ShaderEvaluationCacheStats::default(),
        })
    }

    fn submit(&self, request: ShaderEvaluationWorkerRequest) -> Result<(), String> {
        self.requests
            .send(request)
            .map_err(|_| "shader evaluation worker stopped".to_owned())
    }

    fn receive(&mut self) -> Result<Arc<remu_shader::ShaderEvaluation>, String> {
        let result = self
            .results
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .recv()
            .map_err(|_| "shader evaluation worker stopped".to_owned())?;
        self.stats = result.stats;
        result.evaluation
    }
}

#[derive(Debug)]
struct ShaderEvaluationWorkers {
    parallel: ShaderEvaluationWorker,
}

impl ShaderEvaluationWorkers {
    fn new() -> Result<Self, String> {
        Ok(Self {
            parallel: ShaderEvaluationWorker::spawn("REMu shader evaluator")?,
        })
    }

    fn stats(&self) -> ShaderEvaluationCacheStats {
        self.parallel.stats
    }
}

fn canonical_vertex_record_count(record_count: Option<u32>) -> Option<u32> {
    record_count.map(|count| count.max(1).checked_next_power_of_two().unwrap_or(u32::MAX))
}

fn shader_evaluation_user_data_key(
    state: &remu_shader::ShaderState,
) -> ShaderEvaluationUserDataKey {
    let scalar_base = state.user_data_scalar_register_base;
    let mut words = [0_u32; 64];
    let mut len = 0_usize;
    for user_data_index in state
        .program
        .input_scalar_registers()
        .iter()
        .copied()
        .filter_map(|register| register.checked_sub(scalar_base))
        .filter(|offset| *offset < 64)
    {
        let user_data_index = usize::try_from(user_data_index).unwrap_or(usize::MAX);
        words[len] = state
            .user_data
            .get(user_data_index)
            .copied()
            .unwrap_or_default();
        len += 1;
    }
    ShaderEvaluationUserDataKey {
        words,
        len: u8::try_from(len).expect("hardware user data has at most 64 scalar registers"),
    }
}

fn shader_evaluation_cache_disabled() -> bool {
    static DISABLED: OnceLock<bool> = OnceLock::new();
    *DISABLED.get_or_init(|| {
        std::env::var_os("REMU_DISABLE_SHADER_EVALUATION_CACHE").is_some_and(|value| value == "1")
    })
}

impl ShaderEvaluationCache {
    fn evaluate(
        &mut self,
        state: &remu_shader::ShaderState,
        mut options: remu_shader::EvaluationOptions,
    ) -> Result<Arc<remu_shader::ShaderEvaluation>, String> {
        // Draws using the same resources commonly differ only by a few indices. Binding the next
        // power-of-two record range is still bounded by the guest descriptor, yet lets those draws
        // share one evaluated shader instead of continuously rotating this cache.
        options.required_vertex_record_count =
            canonical_vertex_record_count(options.required_vertex_record_count);
        if shader_evaluation_cache_disabled() {
            self.misses = self.misses.saturating_add(1);
            return remu_shader::evaluate_shader(&ProcessGuestMemory, state, options)
                .map(Arc::new)
                .map_err(|error| {
                    self.failures = self.failures.saturating_add(1);
                    format!("evaluation-error({error})")
                });
        }
        let key = ShaderEvaluationCacheKey {
            program_address: state.program.address,
            // Only scalar inputs actually referenced by the decoded program can affect control
            // flow, resource discovery, or the generated Metal shader. Unity updates unused
            // hardware user-data slots extremely often, so including them here caused false
            // misses and expensive whole-generation cache destruction.
            user_data: shader_evaluation_user_data_key(state),
            compute_system_registers: state.compute_system_registers,
            user_data_scalar_register_base: state.user_data_scalar_register_base,
            options,
        };
        if let Some(cached) = self.entries.get(&key).map(|entry| entry.value.clone()) {
            if cached
                .as_ref()
                .is_ok_and(|evaluation| shader_evaluation_dependencies_are_current(evaluation))
            {
                self.hits = self.hits.saturating_add(1);
                return cached.map_err(|error| error.to_string());
            }
            // The pointer-valued user data can be identical while the guest rewrites the image,
            // sampler, or buffer descriptor behind it. A stale evaluation is worse than a miss:
            // it binds unrelated textures and can turn a complete frame into partial geometry.
            if let Some(retired) = self.entries.remove(&key) {
                retire_shader_evaluation(retired.value);
            }
            self.invalidations = self.invalidations.saturating_add(1);
        }
        self.misses = self.misses.saturating_add(1);
        let evaluated = remu_shader::evaluate_shader(&ProcessGuestMemory, state, options)
            .map(Arc::new)
            .map_err(|error| Arc::<str>::from(format!("evaluation-error({error})")));
        if evaluated.is_err() {
            self.failures = self.failures.saturating_add(1);
        }
        // Most Unity resource states are used by one draw only. Admit only successful states that
        // have been observed twice. Once the cache is full, evict one oldest live admission for
        // each replacement.
        // Whole-generation replacement caused large object graphs to retire together, visibly
        // stalling guest submission and starving audio for several frames.
        // Evaluation failures often describe a transient null descriptor while Unity is
        // populating a command buffer. Never make such a failure persistent for the same pointer
        // state; only successful evaluations participate in the two-hit admission filter.
        let admit = if evaluated.is_ok() {
            if self.seen_once.remove(&key).is_some() {
                true
            } else {
                while self.pending_admission_order.len() >= SHADER_EVALUATION_PENDING_CAPACITY {
                    let Some((old_key, admission_id)) = self.pending_admission_order.pop_front()
                    else {
                        break;
                    };
                    let is_current = self
                        .seen_once
                        .get(&old_key)
                        .is_some_and(|current| *current == admission_id);
                    if is_current {
                        self.seen_once.remove(&old_key);
                    }
                }
                let admission_id = self.next_admission_id;
                self.next_admission_id = self.next_admission_id.wrapping_add(1);
                self.seen_once.insert(key.clone(), admission_id);
                self.pending_admission_order
                    .push_back((key.clone(), admission_id));
                false
            }
        } else {
            false
        };
        if admit {
            self.seen_once.remove(&key);
            while self.entries.len() >= SHADER_EVALUATION_CACHE_CAPACITY {
                let Some((old_key, admission_id)) = self.admission_order.pop_front() else {
                    break;
                };
                let is_current = self
                    .entries
                    .get(&old_key)
                    .is_some_and(|entry| entry.admission_id == admission_id);
                if is_current {
                    if let Some(retired) = self.entries.remove(&old_key) {
                        retire_shader_evaluation(retired.value);
                        self.evictions = self.evictions.saturating_add(1);
                    }
                }
            }
            let admission_id = self.next_admission_id;
            self.next_admission_id = self.next_admission_id.wrapping_add(1);
            self.entries.insert(
                key.clone(),
                ShaderEvaluationCacheEntry {
                    value: evaluated.clone(),
                    admission_id,
                },
            );
            self.admission_order.push_back((key, admission_id));
        }
        evaluated.map_err(|error| error.to_string())
    }
}

fn shader_evaluation_dependencies_are_current(evaluation: &remu_shader::ShaderEvaluation) -> bool {
    #[cfg(target_os = "macos")]
    {
        // Dependencies are address-sorted by the evaluator. Reuse the containing readable range
        // for every following word instead of performing a TLS generation check and binary search
        // for each four-byte scalar load on every cache hit.
        let mut readable_end = 0_u64;
        for &(address, expected) in &evaluation.scalar_memory_reads {
            let Some(word_end) = address.checked_add(4) else {
                return false;
            };
            if word_end > readable_end {
                let Some(mapped_end) = cached_kernel_memory_readable_range_end(address) else {
                    return false;
                };
                if word_end > mapped_end {
                    return false;
                }
                readable_end = mapped_end;
            }
            // SAFETY: guest mapping mutations are serialized by the process dispatch lock. The
            // cached range snapshot proves this identity-mapped word readable for this dispatch.
            let current = unsafe { (address as *const u32).read_unaligned() };
            if u32::from_le(current) != expected {
                return false;
            }
        }
        true
    }
    #[cfg(not(target_os = "macos"))]
    {
        evaluation
            .scalar_memory_reads
            .iter()
            .all(|&(address, expected)| {
                let mut bytes = [0_u8; 4];
                ProcessGuestMemory.read(address, &mut bytes)
                    && u32::from_le_bytes(bytes) == expected
            })
    }
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
    traced_scanout_shader_pairs: BTreeSet<(u64, u64)>,
    traced_target_shader_pairs: BTreeSet<(u64, u64)>,
    traced_sampled_target_shader_pairs: BTreeSet<(u64, u64)>,
    traced_sampled_compute_shaders: BTreeSet<u64>,
    sampled_texture_addresses: BTreeSet<u64>,
    failed_shader_evaluations: BTreeSet<(u8, u64)>,
    known_color_targets: BTreeMap<u64, remu_gpu::GpuColorTarget>,
    shader_evaluation_cache: ShaderEvaluationCache,
    shader_evaluation_workers: Option<ShaderEvaluationWorkers>,
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

const KERNEL_EVENT_FILTER_GRAPHICS: i16 = -14;
const KERNEL_EVENT_FILTER_USER: i16 = -11;
/// Gen5 AP/AMPR command-buffer completion. The Gen4 filter set ends at the high-resolution timer,
/// so the console assigns this newer source the next identifier below it.
const KERNEL_EVENT_FILTER_AMPR: i16 = -16;

static PROBE_DISPATCH: OnceLock<RwLock<ProbeDispatchState>> = OnceLock::new();
static KERNEL_EQUEUE_WAKE: OnceLock<(Mutex<u64>, Condvar)> = OnceLock::new();
static AUDIO_OUT: OnceLock<Mutex<audio::AudioOutState>> = OnceLock::new();
static AUDIO_OUT2: OnceLock<Mutex<AudioOut2State>> = OnceLock::new();
static KERNEL_FILES: OnceLock<Mutex<KernelFileState>> = OnceLock::new();
static PROCESS_CLOCK_EPOCH: OnceLock<Instant> = OnceLock::new();
static PROCESS_START_NANOS: AtomicU64 = AtomicU64::new(0);
static NEXT_NET_POOL_HANDLE: AtomicUsize = AtomicUsize::new(1);
static NEXT_SSL_CONTEXT: AtomicUsize = AtomicUsize::new(1);
static NEXT_HTTP_CONTEXT: AtomicUsize = AtomicUsize::new(1);
static NEXT_NP_WEB_API2_CONTEXT: AtomicUsize = AtomicUsize::new(1);
static DISPATCH_LOCK_STALL_LOGS: AtomicUsize = AtomicUsize::new(0);
static FMOD_CORE_SYSTEM_INITIALIZE_TARGET: AtomicUsize = AtomicUsize::new(0);
static FMOD_STUDIO_SYSTEM_INITIALIZE_TARGET: AtomicUsize = AtomicUsize::new(0);
static GUEST_EXCEPTION_DELIVERY: OnceLock<Mutex<GuestExceptionDeliveryState>> = OnceLock::new();

#[derive(Debug, Default)]
struct AudioOut2State {
    next_handle: u64,
    contexts: BTreeSet<u64>,
    users: BTreeSet<u64>,
    ports: BTreeMap<u64, AudioOut2PortState>,
    attribute_updates: u64,
}

#[derive(Debug)]
struct KernelFileState {
    next_fd: i32,
    files: BTreeMap<i32, std::fs::File>,
    directories: BTreeMap<i32, KernelDirectory>,
    next_file_number: u32,
}

impl Default for KernelFileState {
    fn default() -> Self {
        Self {
            next_fd: 3,
            files: BTreeMap::new(),
            directories: BTreeMap::new(),
            next_file_number: 1,
        }
    }
}

/// A directory opened by the guest, holding its entries already packed as console dirent records.
///
/// The listing is captured when the directory is opened so repeated reads see one stable
/// enumeration, matching a guest that walks a directory while writing files elsewhere.
#[derive(Debug, Default)]
struct KernelDirectory {
    records: Vec<u8>,
    offset: usize,
}

/// Bytes of fixed dirent header before the name: `d_fileno`, `d_reclen`, `d_type`, `d_namlen`.
const DIRENT_HEADER_SIZE: usize = 8;
/// Console dirent records never straddle this boundary, so a reader can consume whole blocks.
const DIRENT_BLOCK_SIZE: usize = 512;
const DIRENT_TYPE_DIRECTORY: u8 = 4;
const DIRENT_TYPE_FILE: u8 = 8;

impl KernelDirectory {
    /// Packs one directory listing into console dirent records.
    ///
    /// Each record is four-byte aligned, and a record that would cross a 512-byte block is pushed
    /// to the next block by extending the previous record's length. Readers rely on that
    /// invariant to consume a block at a time without partially decoding a trailing record.
    fn new(entries: &[(String, bool)], next_file_number: &mut u32) -> Self {
        let mut records: Vec<u8> = Vec::new();
        let mut last_record_start: Option<usize> = None;
        for (name, is_directory) in entries {
            let name_bytes = name.as_bytes();
            let Ok(name_length) = u8::try_from(name_bytes.len()) else {
                // The console's dirent name length is a single byte; longer names cannot be
                // represented and are skipped rather than truncated into a different name.
                continue;
            };
            let record_length = (DIRENT_HEADER_SIZE + name_bytes.len() + 1).next_multiple_of(4);
            let block_remaining = DIRENT_BLOCK_SIZE - records.len() % DIRENT_BLOCK_SIZE;
            if record_length > block_remaining {
                // Extend the previous record over the padding so the block stays fully described.
                if let Some(start) = last_record_start {
                    let previous = u16::from_le_bytes([records[start + 4], records[start + 5]]);
                    let extended =
                        previous.saturating_add(u16::try_from(block_remaining).unwrap_or(u16::MAX));
                    records[start + 4..start + 6].copy_from_slice(&extended.to_le_bytes());
                }
                records.resize(records.len() + block_remaining, 0);
            }
            let file_number = *next_file_number;
            *next_file_number = next_file_number.wrapping_add(1).max(1);
            last_record_start = Some(records.len());
            records.extend_from_slice(&file_number.to_le_bytes());
            records.extend_from_slice(
                &u16::try_from(record_length)
                    .unwrap_or(u16::MAX)
                    .to_le_bytes(),
            );
            records.push(if *is_directory {
                DIRENT_TYPE_DIRECTORY
            } else {
                DIRENT_TYPE_FILE
            });
            records.push(name_length);
            records.extend_from_slice(name_bytes);
            records.resize(last_record_start.unwrap_or(0) + record_length, 0);
        }
        if !records.is_empty() {
            // Pad the final block so every returned read is a whole number of blocks.
            let tail = records.len() % DIRENT_BLOCK_SIZE;
            if tail != 0 {
                let padding = DIRENT_BLOCK_SIZE - tail;
                if let Some(start) = last_record_start {
                    let previous = u16::from_le_bytes([records[start + 4], records[start + 5]]);
                    let extended =
                        previous.saturating_add(u16::try_from(padding).unwrap_or(u16::MAX));
                    records[start + 4..start + 6].copy_from_slice(&extended.to_le_bytes());
                }
                records.resize(records.len() + padding, 0);
            }
        }
        Self { records, offset: 0 }
    }
}

#[derive(Clone, Copy, Debug)]
struct AudioOut2PortState {
    context: u64,
    backend_handle: u64,
    buffer_address: u64,
}

const AUDIO_OUT2_MIX_QUANTUM_FRAMES: u64 = 0x100;
const AUDIO_OUT2_FLOAT_SURROUND_FORMAT: u64 = 5;

impl AudioOut2State {
    fn allocate_handle(&mut self) -> u64 {
        let handle = self.next_handle.max(1);
        self.next_handle = handle.saturating_add(1);
        handle
    }
}
static GUEST_EXCEPTION_PENDING: AtomicBool = AtomicBool::new(false);
#[cfg(target_os = "macos")]
static KERNEL_MEMORY_READABLE_RANGES: OnceLock<RwLock<Arc<[(u64, u64)]>>> = OnceLock::new();
#[cfg(target_os = "macos")]
static KERNEL_MEMORY_READABLE_RANGES_GENERATION: AtomicUsize = AtomicUsize::new(0);

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct PendingGuestException {
    handler: u64,
    exception: i32,
}

#[derive(Debug, Default)]
struct GuestExceptionDeliveryState {
    pending: BTreeMap<u64, PendingGuestException>,
    active: BTreeSet<u64>,
}

fn guest_exception_delivery() -> &'static Mutex<GuestExceptionDeliveryState> {
    GUEST_EXCEPTION_DELIVERY.get_or_init(|| Mutex::new(GuestExceptionDeliveryState::default()))
}

fn guest_exception_is_pending(thread: u64) -> bool {
    thread != 0
        && guest_exception_delivery()
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .pending
            .contains_key(&thread)
}

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
static IMPORT_TELEMETRY_GENERATION: AtomicUsize = AtomicUsize::new(0);
static TRACE_HOT_IMPORTS: OnceLock<bool> = OnceLock::new();

#[derive(Debug, Default)]
struct GuestSyncState {
    rwlocks: HashMap<u64, GuestRwlockState>,
    conditions: HashMap<u64, GuestConditionState>,
    condition_waiters: HashMap<u64, GuestConditionWaiterState>,
    next_condition_waiter: u64,
    condition_notify_count: u64,
    semaphores: HashMap<u32, GuestSemaphoreState>,
    next_semaphore_handle: u32,
    semaphore_wait_count: u64,
    semaphore_signal_count: u64,
    event_flags: HashMap<u64, GuestEventFlagState>,
    next_event_flag_handle: u64,
}

#[derive(Debug, Default)]
struct GuestConditionState {
    waiters: VecDeque<u64>,
    bound_mutex: Option<u64>,
}

impl GuestConditionState {
    fn register_waiter(&mut self, waiter: u64, mutex: u64) -> bool {
        if self.bound_mutex.is_some_and(|bound| bound != mutex) {
            return false;
        }
        self.bound_mutex = Some(mutex);
        self.waiters.push_back(waiter);
        true
    }

    fn release_one(&mut self) -> Option<u64> {
        let waiter = self.waiters.pop_front();
        self.clear_binding_if_idle();
        waiter
    }

    fn release_all(&mut self) -> VecDeque<u64> {
        self.bound_mutex = None;
        std::mem::take(&mut self.waiters)
    }

    fn cancel_wait(&mut self, waiter: u64) {
        self.waiters.retain(|candidate| *candidate != waiter);
        self.clear_binding_if_idle();
    }

    fn is_idle(&self) -> bool {
        self.waiters.is_empty()
    }

    fn clear_binding_if_idle(&mut self) {
        if self.waiters.is_empty() {
            self.bound_mutex = None;
        }
    }
}

#[derive(Debug)]
struct GuestConditionWaiterState {
    released: bool,
}

impl GuestSyncState {
    fn register_condition_waiter(&mut self, condition: u64, mutex: u64) -> Option<u64> {
        if self
            .conditions
            .get(&condition)
            .and_then(|condition| condition.bound_mutex)
            .is_some_and(|bound| bound != mutex)
        {
            return None;
        }
        let waiter = loop {
            self.next_condition_waiter = self.next_condition_waiter.wrapping_add(1).max(1);
            if !self
                .condition_waiters
                .contains_key(&self.next_condition_waiter)
            {
                break self.next_condition_waiter;
            }
        };
        assert!(
            self.conditions
                .entry(condition)
                .or_default()
                .register_waiter(waiter, mutex),
            "condition mutex binding was validated under the synchronization lock"
        );
        self.condition_waiters
            .insert(waiter, GuestConditionWaiterState { released: false });
        Some(waiter)
    }

    fn condition_waiter_released(&self, waiter: u64) -> bool {
        self.condition_waiters
            .get(&waiter)
            .is_some_and(|waiter| waiter.released)
    }

    fn finish_condition_wait(&mut self, condition: u64, waiter: u64, timed_out: bool) {
        if timed_out {
            self.conditions
                .get_mut(&condition)
                .expect("a pending condition waiter prevents destruction")
                .cancel_wait(waiter);
        }
        let waiter_state = self
            .condition_waiters
            .remove(&waiter)
            .expect("registered condition waiter survives condition destruction");
        assert_eq!(
            waiter_state.released, !timed_out,
            "timeout and notification have exactly one winner"
        );
    }
}

#[derive(Debug)]
struct GuestMutexState {
    owner: Option<ThreadId>,
    recursion: u32,
    mutex_type: i32,
    waiters: u32,
}

#[derive(Debug)]
struct GuestMutexEntry {
    state: Mutex<GuestMutexState>,
    wake: Condvar,
}

#[derive(Debug, Default)]
struct GuestRwlockState {
    readers: HashMap<ThreadId, u32>,
    reader_total: u32,
    writer: Option<ThreadId>,
    waiting_writers: u32,
}

impl Default for GuestMutexState {
    fn default() -> Self {
        let attributes = PthreadMutexAttrState::default();
        Self {
            owner: None,
            recursion: 0,
            mutex_type: attributes.mutex_type,
            waiters: 0,
        }
    }
}

#[derive(Debug)]
struct GuestSemaphoreState {
    count: i32,
    maximum: i32,
}

#[derive(Debug)]
struct GuestEventFlagState {
    name: String,
    attributes: u32,
    bits: u64,
    waiters: u32,
    cancellation_generation: u64,
}

static GUEST_SYNC: OnceLock<(Mutex<GuestSyncState>, Condvar)> = OnceLock::new();
const GUEST_MUTEX_SHARD_COUNT: usize = 64;
static GUEST_MUTEX_SHARDS: OnceLock<
    [Mutex<HashMap<u64, Arc<GuestMutexEntry>>>; GUEST_MUTEX_SHARD_COUNT],
> = OnceLock::new();

thread_local! {
    static CURRENT_GUEST_THREAD_POINTER: Cell<u64> = const { Cell::new(0) };
    static CURRENT_GUEST_THREAD_HANDLE: Cell<u64> = const { Cell::new(0) };
    static CURRENT_GUEST_THREAD_EXIT_CONTEXT: Cell<*mut GuestThreadExitContext> = const { Cell::new(std::ptr::null_mut()) };
    static GUEST_PTHREAD_SPECIFIC: RefCell<BTreeMap<u32, u64>> = const { RefCell::new(BTreeMap::new()) };
    static IMPORT_TELEMETRY_CACHE: RefCell<Option<(usize, Arc<ImportTelemetry>)>> = const { RefCell::new(None) };
    static PROCESS_TIME_CALLSITES: RefCell<[(u64, u64); 8]> = const { RefCell::new([(0, 0); 8]) };
    #[cfg(target_os = "macos")]
    static KERNEL_MEMORY_READABLE_RANGES_CACHE: RefCell<Option<(usize, Arc<[(u64, u64)]>)>> = const { RefCell::new(None) };
}

#[cfg(target_os = "macos")]
fn cached_kernel_memory_readable_range_end(address: u64) -> Option<u64> {
    let generation = KERNEL_MEMORY_READABLE_RANGES_GENERATION.load(Ordering::Acquire);
    KERNEL_MEMORY_READABLE_RANGES_CACHE.with(|cache| {
        let mut cache = cache.borrow_mut();
        if cache
            .as_ref()
            .is_none_or(|(cached_generation, _)| *cached_generation != generation)
        {
            let ranges = KERNEL_MEMORY_READABLE_RANGES
                .get_or_init(|| RwLock::new(Arc::from([])))
                .read()
                .unwrap_or_else(std::sync::PoisonError::into_inner)
                .clone();
            *cache = Some((generation, ranges));
        }
        let ranges = &cache
            .as_ref()
            .expect("readable range cache was populated")
            .1;
        let index = ranges.partition_point(|&(start, _)| start <= address);
        index
            .checked_sub(1)
            .and_then(|index| ranges.get(index))
            .filter(|&&(start, end)| address >= start && address < end)
            .map(|&(_, end)| end)
    })
}

fn cached_import_telemetry() -> *const ImportTelemetry {
    let generation = IMPORT_TELEMETRY_GENERATION.load(Ordering::Acquire);
    IMPORT_TELEMETRY_CACHE.with(|cache| {
        let mut cache = cache.borrow_mut();
        if cache
            .as_ref()
            .is_none_or(|(cached_generation, _)| *cached_generation != generation)
        {
            let telemetry = IMPORT_TELEMETRY
                .get_or_init(|| RwLock::new(Arc::new(ImportTelemetry::new(Vec::new(), 0))))
                .read()
                .unwrap_or_else(std::sync::PoisonError::into_inner)
                .clone();
            *cache = Some((generation, telemetry));
        }
        Arc::as_ptr(&cache.as_ref().expect("telemetry cache was populated").1)
    })
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

#[repr(C)]
#[derive(Debug, Default)]
struct GuestThreadExitContext {
    stack_pointer: u64,
    continuation: u64,
}

/// Calls a guest entry behind a stack boundary that `scePthreadExit` can return to.
///
/// Guest pthread implementations terminate from arbitrary depths and deliberately place a trap
/// after their non-returning exit import. Calling host `pthread_exit` would bypass Rust's thread
/// closure and abort the process. This boundary instead restores its own callee-saved frame and
/// lets the Rust wrapper complete normally with the guest-supplied return value.
#[cfg(target_arch = "x86_64")]
#[unsafe(no_mangle)]
#[unsafe(naked)]
unsafe extern "C" fn invoke_guest_entry_with_exit_boundary(
    _entry: u64,
    _argument0: u64,
    _argument1: u64,
    _context: *mut GuestThreadExitContext,
) -> u64 {
    core::arch::naked_asm!(
        "push rbp",
        "mov rbp, rsp",
        "push rbx",
        "push r12",
        "push r13",
        "push r14",
        "push r15",
        "sub rsp, 8",
        "mov [rcx], rsp",
        "lea rax, [rip + 2f]",
        "mov [rcx + 8], rax",
        "mov rax, rdi",
        "mov rdi, rsi",
        "mov rsi, rdx",
        "call rax",
        "2:",
        "add rsp, 8",
        "pop r15",
        "pop r14",
        "pop r13",
        "pop r12",
        "pop rbx",
        "pop rbp",
        "ret",
    );
}

#[cfg(target_arch = "x86_64")]
#[unsafe(no_mangle)]
#[unsafe(naked)]
unsafe extern "C" fn escape_guest_entry(_context: *const GuestThreadExitContext, _value: u64) -> ! {
    core::arch::naked_asm!("mov rsp, [rdi]", "mov rax, rsi", "jmp [rdi + 8]",);
}

#[cfg(target_arch = "x86_64")]
fn invoke_guest_entry_with_exit(entry: u64, argument0: u64, argument1: u64) -> u64 {
    let mut context = GuestThreadExitContext::default();
    let previous = CURRENT_GUEST_THREAD_EXIT_CONTEXT.replace(&raw mut context);
    // SAFETY: the mapped guest entry uses the System V ABI. The assembly boundary preserves every
    // callee-saved register and records a continuation in this live stack-owned context.
    let result = unsafe {
        invoke_guest_entry_with_exit_boundary(entry, argument0, argument1, &raw mut context)
    };
    CURRENT_GUEST_THREAD_EXIT_CONTEXT.set(previous);
    result
}

#[cfg(target_arch = "x86_64")]
fn guest_pthread_exit(value: u64) -> ! {
    let context = CURRENT_GUEST_THREAD_EXIT_CONTEXT.get();
    if context.is_null() {
        eprintln!("[REMu][THREAD] scePthreadExit called outside a guest thread boundary");
        terminate_probe(87);
    }
    // SAFETY: the context belongs to this native thread and remains live until its guest entry
    // either returns or exits. This import is handled before acquiring process-wide locks, so the
    // non-local escape cannot strand a Rust synchronization guard.
    unsafe { escape_guest_entry(context, value) }
}

#[cfg(not(target_arch = "x86_64"))]
fn guest_pthread_exit(_value: u64) -> ! {
    unreachable!("guest pthread execution requires an x86-64 host process")
}

#[cfg(not(target_arch = "x86_64"))]
fn invoke_guest_entry_with_exit(_entry: u64, _argument0: u64, _argument1: u64) -> u64 {
    unreachable!("guest pthread execution requires an x86-64 host process")
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
    IMPORT_TELEMETRY_GENERATION.fetch_add(1, Ordering::Release);
    state.unimplemented_nids.clear();
    #[cfg(target_os = "macos")]
    {
        state.kernel_memory = None;
        *KERNEL_MEMORY_READABLE_RANGES
            .get_or_init(|| RwLock::new(Arc::from([])))
            .write()
            .unwrap_or_else(std::sync::PoisonError::into_inner) = Arc::from([]);
        KERNEL_MEMORY_READABLE_RANGES_GENERATION.fetch_add(1, Ordering::Release);
        state.process_mapping_start = image.memory.mapping_start().get();
        state.process_mapping_end = state
            .process_mapping_start
            .checked_add(image.memory.mapping_size())
            .expect("the validated process mapping range cannot overflow");
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
    state.bundled_modules.clone_from(&image.bundled_modules);
    state.unwind_modules.clone_from(&image.unwind_modules);
    state.installed_exception_handlers.clear();
    state.next_ajm_context = 1;
    state.ajm_contexts.clear();
    *guest_exception_delivery()
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner) =
        GuestExceptionDeliveryState::default();
    GUEST_EXCEPTION_PENDING.store(false, Ordering::Release);
    state.heap_trace_storage.clear();
    state.heap_trace_storage.resize(65, 0);
    state.guest_mspaces.clear();
    state.json_runtime = JsonRuntimeState::default();
    state.av_player = AvPlayerRuntimeState::default();
    state.services = RuntimeServiceState::default();
    reset_kernel_process_time();
    *AUDIO_OUT
        .get_or_init(|| Mutex::new(audio::AudioOutState::default()))
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner) = audio::AudioOutState::default();
    *AUDIO_OUT2
        .get_or_init(|| Mutex::new(AudioOut2State::default()))
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner) = AudioOut2State::default();
    FMOD_STUDIO_SYSTEM_INITIALIZE_TARGET.store(0, Ordering::Release);
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
    state.next_equeue = 1;
    state.kernel_equeues.clear();
    state.kernel_event_registrations.clear();
    state.pending_events.clear();
    state.pending_agc_completions.clear();
    *KERNEL_FILES
        .get_or_init(|| Mutex::new(KernelFileState::default()))
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner) = KernelFileState::default();
    state.ampr_files.clear();
    state.ampr_file_handles.clear();
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

#[repr(C, align(16))]
struct GuestExceptionContext([u8; 0x500]);

fn write_exception_context_word(context: &mut GuestExceptionContext, offset: usize, value: u64) {
    context.0[offset..offset + 8].copy_from_slice(&value.to_le_bytes());
}

fn deliver_pending_guest_exception_at_import(frame: &ImportCallFrame) {
    let thread = current_guest_thread_handle();
    if thread == 0 || (!cfg!(test) && !GUEST_EXCEPTION_PENDING.load(Ordering::Acquire)) {
        return;
    }
    let pending = {
        let mut delivery = guest_exception_delivery()
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        if delivery.active.contains(&thread) {
            return;
        }
        let Some(pending) = delivery.pending.remove(&thread) else {
            return;
        };
        GUEST_EXCEPTION_PENDING.store(!delivery.pending.is_empty(), Ordering::Release);
        delivery.active.insert(thread);
        pending
    };

    let mut context = GuestExceptionContext([0; 0x500]);
    const MCONTEXT: usize = 0x40;
    for (offset, value) in [
        (0x08, frame.integer_arguments[0]),
        (0x10, frame.integer_arguments[1]),
        (0x18, frame.integer_arguments[2]),
        (0x20, frame.integer_arguments[3]),
        (0x28, frame.integer_arguments[4]),
        (0x30, frame.integer_arguments[5]),
        (0x38, frame.rax),
    ] {
        write_exception_context_word(&mut context, MCONTEXT + offset, value);
    }
    write_exception_context_word(&mut context, MCONTEXT + 0xa0, frame.guest_return_address);
    write_exception_context_word(&mut context, MCONTEXT + 0xb8, frame.guest_rsp);
    write_exception_context_word(&mut context, MCONTEXT + 0xc8, 0x480);
    write_exception_context_word(
        &mut context,
        MCONTEXT + 0x440,
        current_guest_thread_pointer(),
    );

    eprintln!(
        "[REMu][KERNEL] delivering cooperative exception target={thread:#x} type={} handler={:#x} rip={:#x} rsp={:#x}",
        pending.exception, pending.handler, frame.guest_return_address, frame.guest_rsp,
    );
    type GuestExceptionHandler = unsafe extern "C" fn(i32, *mut u8);
    // SAFETY: The process installed this mapped executable guest handler for the requested signal.
    // Delivery runs at an import boundary on the target's own native thread, so its guest TLS and
    // stack identity remain intact while the handler observes the synthesized Orbis ucontext.
    let handler: GuestExceptionHandler = unsafe { std::mem::transmute(pending.handler) };
    // SAFETY: `context` is live and writable for the complete synchronous handler invocation.
    unsafe { handler(pending.exception, context.0.as_mut_ptr()) };

    guest_exception_delivery()
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .active
        .remove(&thread);
}

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
        dispatch = sym import_dispatch_entry,
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
        import_dispatch_entry as extern "C" fn(*const ImportCallFrame) -> ProbeImportReturn,
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
/// Number of leading guest imports to trace with their return values.
///
/// Boot-order defects are easier to read as an exact call/return sequence than as first-hit
/// records, because a wrong result only becomes visible in the guest's next branch. The trace is
/// bounded so it cannot dominate a long run.
fn import_call_trace_limit() -> u64 {
    static LIMIT: OnceLock<u64> = OnceLock::new();
    *LIMIT.get_or_init(|| {
        std::env::var("REMU_TRACE_IMPORT_CALLS")
            .ok()
            .and_then(|value| value.parse::<u64>().ok())
            .unwrap_or(0)
    })
}

static IMPORT_CALL_TRACE_SEQUENCE: AtomicU64 = AtomicU64::new(0);

extern "C" fn import_dispatch_entry(frame: *const ImportCallFrame) -> ProbeImportReturn {
    let limit = import_call_trace_limit();
    if limit == 0 {
        return import_dispatch(frame);
    }
    let sequence = IMPORT_CALL_TRACE_SEQUENCE.fetch_add(1, Ordering::Relaxed) + 1;
    if sequence > limit || frame.is_null() {
        return import_dispatch(frame);
    }
    // SAFETY: the caller supplies one aligned, fully initialized frame that stays live for the
    // complete call, and the null case returned above.
    let (index, arguments, guest_return_address) = unsafe {
        (
            (*frame).dispatch_index,
            (*frame).integer_arguments,
            (*frame).guest_return_address,
        )
    };
    // SAFETY: the telemetry Arc is retained by this thread's cache for the whole call.
    let telemetry = unsafe { &*cached_import_telemetry() };
    let nid = usize::try_from(index)
        .ok()
        .and_then(|index| telemetry.nids.get(index))
        .map_or("?", String::as_str);
    let name = known_nid_name(nid).unwrap_or("unknown");
    eprintln!(
        "[REMu][CALL] seq={sequence} nid={nid} name={name} thread={} return={guest_return_address:#x} args={:#x},{:#x},{:#x},{:#x},{:#x},{:#x}",
        current_guest_thread_handle(),
        arguments[0],
        arguments[1],
        arguments[2],
        arguments[3],
        arguments[4],
        arguments[5],
    );
    let result = import_dispatch(frame);
    eprintln!(
        "[REMu][CALL] seq={sequence} nid={nid} name={name} result={:#x}",
        result.integer
    );
    result
}

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
    deliver_pending_guest_exception_at_import(frame);
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
    // `configure_probe_dispatch` replaces telemetry only before guest threads start. The current
    // thread retains the Arc in `IMPORT_TELEMETRY_CACHE` for the complete call, so this pointer
    // remains live while avoiding a contended RwLock and atomic Arc clone on every guest import.
    let telemetry = unsafe { &*cached_import_telemetry() };
    let Some(index) = usize::try_from(index).ok() else {
        eprintln!("[REMu][IMPORT] invalid stub index {index}");
        terminate_probe(87);
    };
    let Some(nid) = telemetry.nids.get(index).map(String::as_str) else {
        eprintln!("[REMu][IMPORT] invalid stub index {index}");
        terminate_probe(87);
    };
    if matches!(nid, "3kg7rT0NQIs" | "FJrT5LuUBAU") {
        guest_pthread_exit(call_args[0]);
    }
    let hit = telemetry.hits.fetch_add(1, Ordering::Relaxed) + 1;
    telemetry.counts[index].fetch_add(1, Ordering::Relaxed);
    if nid == "fgxnMeTNUtY"
        && *TRACE_HOT_IMPORTS.get_or_init(|| {
            std::env::var_os("REMU_TRACE_HOT_IMPORTS").is_some_and(|value| value == "1")
        })
    {
        PROCESS_TIME_CALLSITES.with(|callsites| {
            let mut callsites = callsites.borrow_mut();
            let slot = callsites
                .iter()
                .position(|(address, _)| *address == guest_return_address)
                .or_else(|| callsites.iter().position(|(address, _)| *address == 0))
                .unwrap_or_else(|| {
                    callsites
                        .iter()
                        .enumerate()
                        .min_by_key(|(_, (_, count))| *count)
                        .map_or(0, |(index, _)| index)
                });
            if callsites[slot].0 != guest_return_address {
                callsites[slot] = (guest_return_address, 0);
            }
            callsites[slot].1 = callsites[slot].1.saturating_add(1);
            let count = callsites[slot].1;
            if count >= 1_024 && count.is_power_of_two() {
                eprintln!(
                    "[REMu][IMPORT][TIMER] return={guest_return_address:#x} calls={count} rdi={:#x} rsi={:#x}",
                    call_args[0], call_args[1]
                );
            }
        });
    }
    if hit >= (1 << 20)
        && hit.is_power_of_two()
        && *TRACE_HOT_IMPORTS.get_or_init(|| {
            std::env::var_os("REMU_TRACE_HOT_IMPORTS").is_some_and(|value| value == "1")
        })
    {
        let summary = telemetry
            .hot_imports(8)
            .into_iter()
            .map(|(nid, count)| {
                format!(
                    "{}:{}={count}",
                    known_nid_name(nid).unwrap_or("unknown"),
                    nid
                )
            })
            .collect::<Vec<_>>()
            .join(",");
        eprintln!("[REMu][IMPORT] hits={hit} hot=[{summary}]");
    }
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
    if matches!(nid, "4J2sUJmuHZQ" | "fgxnMeTNUtY" | "BNowx2l588E") {
        // Process timers are called by FMOD's mixer and streaming workers. They must not queue
        // behind graphics submission on the process-wide dispatch lock or audio production can
        // stall for an entire frame even though reading the monotonic clock is independent.
        let result = match nid {
            "4J2sUJmuHZQ" => kernel_process_time(false),
            "fgxnMeTNUtY" => kernel_process_time(true),
            "BNowx2l588E" => 1_000_000_000,
            _ => unreachable!("the process-timer NID set is exhaustive"),
        };
        return ProbeImportReturn::integer(result);
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
        let result =
            dispatch_guest_sync_import_at_import(nid, call_args, mutex_attributes, Some(frame));
        // A kernel exception can interrupt a thread parked inside a synchronization import. The
        // wait returns as a legal spurious wake, then the target runs its handler on this exact
        // native thread before control crosses back into guest code.
        deliver_pending_guest_exception_at_import(frame);
        return ProbeImportReturn::integer(result);
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
    if matches!(
        nid,
        "JfEPXVxhFqA"
            | "ekNvsT22rsY"
            | "s1--uE9mBFw"
            | "b+uAV89IlxE"
            | "QOQtbeDqsT4"
            | "GrQ9s4IrNaQ"
            | "g2tViFIohHE"
            | "t5YrizufpQc"
            | "pDmme7Bgm6E"
            | "0x6o1VVAYSY"
            | "xywYcRB7nbQ"
            | "JK2wamZPzwM"
            | "IaZXJ9M79uo"
            | "cd+Rtw+D1x8"
            | "on6ZH7Abo10"
            | "8XTArSPyWHk"
            | "PE2zHMqLSHs"
            | "aII9h5nli9U"
    ) {
        return ProbeImportReturn::integer(dispatch_audio_out_import(nid, call_args));
    }
    if let Some(result) = dispatch_stateless_import(nid, call_args, stack_args) {
        return ProbeImportReturn::integer(result);
    }
    if let Some(result) = dispatch_kernel_fd_import(nid, call_args) {
        return ProbeImportReturn::integer(result);
    }
    if nid == "fzyMKs9kim0" {
        // An untimed equeue wait must park without retaining the process-wide dispatch lock.
        // Graphics submission, flip, and completion producers all need that lock briefly to
        // publish their events; holding it here would create the exact producer/consumer
        // deadlock the wait is meant to resolve.
        let result = kernel_wait_equeue_blocking(call_args);
        deliver_pending_guest_exception_at_import(frame);
        return ProbeImportReturn::integer(result);
    }
    if nid == "wzvqT4UqKX8" {
        // Module initialization can itself call guest imports. Resolve and mark the object while
        // holding the process lock, then invoke its DT_INIT entries after releasing that lock.
        return ProbeImportReturn::integer(kernel_load_start_module(call_args));
    }
    if matches!(nid, "o3+RWnHViSg" | "JdksQu8pNdQ") {
        prepare_av_player_video_buffers(call_args[0]);
    }
    let state = PROBE_DISPATCH.get_or_init(|| RwLock::new(ProbeDispatchState::default()));
    let trace_lock_stall =
        std::env::var_os("REMU_TRACE_DISPATCH_STALLS").is_some_and(|value| value == "1");
    let lock_started = trace_lock_stall.then(Instant::now);
    let mut state = state
        .write()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    if let Some(lock_started) = lock_started {
        let waited = lock_started.elapsed();
        if waited >= Duration::from_millis(2)
            && DISPATCH_LOCK_STALL_LOGS.fetch_add(1, Ordering::Relaxed) < 64
        {
            eprintln!(
                "[REMu][IMPORT][LOCK] waited={:.3}ms nid={nid} name={} thread={:?} guest={:#x}",
                waited.as_secs_f64() * 1_000.0,
                known_nid_name(nid).unwrap_or("unknown"),
                std::thread::current().name().unwrap_or("unnamed"),
                current_guest_thread_handle(),
            );
        }
    }
    if nid == "H2e8t5ScQGc" {
        let callbacks = take_exit_callbacks(&mut state, Some(call_args[0]));
        drop(state);
        invoke_guest_exit_callbacks(callbacks);
        return ProbeImportReturn::integer(0);
    }
    if matches!(nid, "6UgtwV+0zb4" | "OxhIB8LB-PQ" | "Jmi+9w9u0E4") {
        if call_args[0] == 0 || call_args[2] == 0 {
            return ProbeImportReturn::integer(sce_kernel_error(SCE_KERNEL_ERROR_INVALID_ARGUMENT));
        }
        let attributes = pthread_attr_state(call_args[1], &state).unwrap_or_default();
        let entry = call_args[2];
        let thread_argument = call_args[3];
        let guest_tls = clone_guest_tls(&state.tls_template);
        // The plain POSIX entry point has no name argument; both named forms pass it fifth.
        let thread_name = matches!(nid, "6UgtwV+0zb4" | "Jmi+9w9u0E4")
            .then(|| read_guest_cstring(call_args[4]))
            .flatten()
            .filter(|name| !name.is_empty());
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
            let (inner_entry, inner_argument) =
                if process_guest_range_is_readable(thread_argument, 2 * std::mem::size_of::<u64>())
                {
                    (
                        read_guest_u64(thread_argument),
                        read_guest_u64(thread_argument + 8),
                    )
                } else {
                    (0, 0)
                };
            eprintln!(
                "[REMu][THREAD] create handle={handle} name={:?} entry={entry:#x} arg={thread_argument:#x} inner={inner_entry:#x}/{inner_argument:#x} detach={} stack={:#x}",
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
    if matches!(nid, "onNY9Byn-W8" | "h9CcP3J0oVM") {
        let posix = nid == "h9CcP3J0oVM";
        if call_args[0] == 0 || state.detached_threads.contains(&call_args[0]) {
            return ProbeImportReturn::integer(join_result(
                sce_kernel_error(SCE_KERNEL_ERROR_INVALID_ARGUMENT),
                posix,
            ));
        }
        let thread_id = call_args[0];
        let handle = state.thread_handles.remove(&thread_id);
        let Some(handle) = handle else {
            return ProbeImportReturn::integer(join_result(
                sce_kernel_error(SCE_KERNEL_ERROR_NOT_FOUND),
                posix,
            ));
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
    if matches!(nid, "4qGrR6eoP9Y" | "+U1R4WtXvoc") {
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
    let result = dispatch_probe_import(nid, call_args, stack_args, &mut state);
    let av_player_callbacks = std::mem::take(&mut state.av_player.pending_callbacks);
    drop(state);
    invoke_av_player_event_callbacks(av_player_callbacks);
    ProbeImportReturn::integer(result)
}

fn dispatch_audio_out_import(nid: &str, arguments: [u64; 6]) -> u64 {
    if matches!(
        nid,
        "g2tViFIohHE"
            | "t5YrizufpQc"
            | "pDmme7Bgm6E"
            | "0x6o1VVAYSY"
            | "xywYcRB7nbQ"
            | "JK2wamZPzwM"
            | "IaZXJ9M79uo"
            | "cd+Rtw+D1x8"
            | "on6ZH7Abo10"
            | "8XTArSPyWHk"
            | "PE2zHMqLSHs"
            | "aII9h5nli9U"
    ) {
        if nid == "g2tViFIohHE" {
            AUDIO_OUT
                .get_or_init(|| Mutex::new(audio::AudioOutState::default()))
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner)
                .initialize();
        }
        return dispatch_audio_out2_import(nid, arguments);
    }

    let state = AUDIO_OUT.get_or_init(|| Mutex::new(audio::AudioOutState::default()));
    let mut state = state
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    match nid {
        "JfEPXVxhFqA" => {
            state.initialize();
            0
        }
        "ekNvsT22rsY" => state.open(arguments).unwrap_or_else(audio_out_error),
        "s1--uE9mBFw" => state
            .close(arguments[0])
            .map_or_else(audio_out_error, |()| 0),
        "b+uAV89IlxE" => state
            .set_volume(arguments)
            .map_or_else(audio_out_error, |()| 0),
        "GrQ9s4IrNaQ" => state
            .get_port_state(arguments)
            .map_or_else(audio_out_error, |()| 0),
        "QOQtbeDqsT4" => {
            let output = state.output(arguments);
            drop(state);
            match output {
                Ok(delay) => {
                    if !delay.is_zero() {
                        std::thread::sleep(delay);
                    }
                    0
                }
                Err(error) => audio_out_error(error),
            }
        }
        _ => unreachable!("caller filters audio-out imports"),
    }
}

fn dispatch_audio_out2_import(nid: &str, arguments: [u64; 6]) -> u64 {
    let state = AUDIO_OUT2.get_or_init(|| Mutex::new(AudioOut2State::default()));
    let mut state = state
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    match nid {
        "g2tViFIohHE" => {
            eprintln!("[REMu][AUDIO2] initialized parameter={:#x}", arguments[0]);
            0
        }
        "t5YrizufpQc" => {
            if arguments[0] == 0 {
                return sce_kernel_error(SCE_KERNEL_ERROR_INVALID_ARGUMENT);
            }
            0
        }
        "pDmme7Bgm6E" => {
            if arguments[0] == 0 || arguments[1] == 0 {
                return sce_kernel_error(SCE_KERNEL_ERROR_INVALID_ARGUMENT);
            }
            const CONTEXT_MEMORY_SIZE: u64 = 0x1_0000;
            write_guest_u64(arguments[1], CONTEXT_MEMORY_SIZE);
            eprintln!(
                "[REMu][AUDIO2] context memory query parameter={:#x} size={CONTEXT_MEMORY_SIZE:#x}",
                arguments[0]
            );
            0
        }
        "0x6o1VVAYSY" => {
            if arguments[0] == 0 || arguments[1] == 0 || arguments[2] == 0 || arguments[3] == 0 {
                return sce_kernel_error(SCE_KERNEL_ERROR_INVALID_ARGUMENT);
            }
            let handle = state.allocate_handle();
            state.contexts.insert(handle);
            write_guest_u64(arguments[3], handle);
            eprintln!(
                "[REMu][AUDIO2] created context={handle} memory={:#x} size={:#x}",
                arguments[1], arguments[2]
            );
            0
        }
        "xywYcRB7nbQ" => {
            if arguments[1] == 0 {
                return sce_kernel_error(SCE_KERNEL_ERROR_INVALID_ARGUMENT);
            }
            let handle = state.allocate_handle();
            state.users.insert(handle);
            write_guest_u64(arguments[1], handle);
            eprintln!(
                "[REMu][AUDIO2] created user={handle} user_id={:#x}",
                arguments[0]
            );
            0
        }
        "JK2wamZPzwM" => {
            if !state.contexts.contains(&arguments[0]) || arguments[1] == 0 || arguments[2] == 0 {
                return sce_kernel_error(SCE_KERNEL_ERROR_INVALID_ARGUMENT);
            }
            let frequency = u64::from(read_guest_u32(arguments[1] + 8));
            let backend_handle = {
                let mut audio = AUDIO_OUT
                    .get_or_init(|| Mutex::new(audio::AudioOutState::default()))
                    .lock()
                    .unwrap_or_else(std::sync::PoisonError::into_inner);
                audio.initialize();
                match audio.open([
                    0,
                    0,
                    0,
                    AUDIO_OUT2_MIX_QUANTUM_FRAMES,
                    frequency,
                    AUDIO_OUT2_FLOAT_SURROUND_FORMAT,
                ]) {
                    Ok(handle) => handle,
                    Err(error) => return audio_out_error(error),
                }
            };
            let handle = state.allocate_handle();
            state.ports.insert(
                handle,
                AudioOut2PortState {
                    context: arguments[0],
                    backend_handle,
                    buffer_address: 0,
                },
            );
            write_guest_u64(arguments[2], handle);
            eprintln!(
                "[REMu][AUDIO2] created port={handle} context={} parameter={:#x} backend={backend_handle} rate={frequency}",
                arguments[0], arguments[1],
            );
            0
        }
        "IaZXJ9M79uo" => {
            state.users.remove(&arguments[0]);
            0
        }
        "cd+Rtw+D1x8" => {
            if let Some(port) = state.ports.remove(&arguments[0]) {
                let _ = AUDIO_OUT
                    .get_or_init(|| Mutex::new(audio::AudioOutState::default()))
                    .lock()
                    .unwrap_or_else(std::sync::PoisonError::into_inner)
                    .close(port.backend_handle);
            }
            0
        }
        "on6ZH7Abo10" => {
            state.contexts.remove(&arguments[0]);
            let backend_handles = state
                .ports
                .extract_if(.., |_, port| port.context == arguments[0])
                .map(|(_, port)| port.backend_handle)
                .collect::<Vec<_>>();
            let mut audio = AUDIO_OUT
                .get_or_init(|| Mutex::new(audio::AudioOutState::default()))
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            for handle in backend_handles {
                let _ = audio.close(handle);
            }
            0
        }
        "8XTArSPyWHk" => {
            if !state.ports.contains_key(&arguments[0]) || arguments[1] == 0 {
                return sce_kernel_error(SCE_KERNEL_ERROR_INVALID_ARGUMENT);
            }
            state.attribute_updates = state.attribute_updates.saturating_add(1);
            let count = arguments[2].min(16);
            for index in 0..count {
                let base = arguments[1] + index * 24;
                let attribute = read_guest_u64(base);
                let data = read_guest_u64(base + 8);
                let size = read_guest_u64(base + 16);
                if attribute == 0
                    && data != 0
                    && size >= 8
                    && let Some(port) = state.ports.get_mut(&arguments[0])
                {
                    port.buffer_address = read_guest_u64(data);
                }
            }
            if state.attribute_updates <= 8 || state.attribute_updates.is_power_of_two() {
                let attributes = (0..count)
                    .map(|index| {
                        let base = arguments[1] + index * 24;
                        let attribute = read_guest_u64(base);
                        let data = read_guest_u64(base + 8);
                        let size = read_guest_u64(base + 16);
                        let value = if data != 0 && size >= 8 {
                            read_guest_u64(data)
                        } else {
                            0
                        };
                        format!("id={attribute:#x}/data={data:#x}/size={size:#x}/value={value:#x}")
                    })
                    .collect::<Vec<_>>()
                    .join(",");
                eprintln!(
                    "[REMu][AUDIO2] attributes update={} port={} count={} [{attributes}]",
                    state.attribute_updates, arguments[0], arguments[2]
                );
            }
            0
        }
        "PE2zHMqLSHs" => {
            if !state.contexts.contains(&arguments[0]) {
                return sce_kernel_error(SCE_KERNEL_ERROR_INVALID_ARGUMENT);
            }
            0
        }
        "aII9h5nli9U" => {
            if !state.contexts.contains(&arguments[0]) {
                return sce_kernel_error(SCE_KERNEL_ERROR_INVALID_ARGUMENT);
            }
            let buffers = state
                .ports
                .values()
                .filter(|port| port.context == arguments[0] && port.buffer_address != 0)
                .map(|port| (port.backend_handle, port.buffer_address))
                .collect::<Vec<_>>();
            drop(state);

            let mut maximum_delay = Duration::ZERO;
            let mut audio = AUDIO_OUT
                .get_or_init(|| Mutex::new(audio::AudioOutState::default()))
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            for (handle, buffer) in buffers {
                match audio.output([handle, buffer, 0, 0, 0, 0]) {
                    Ok(delay) => maximum_delay = maximum_delay.max(delay),
                    Err(error) => return audio_out_error(error),
                }
            }
            drop(audio);
            if !maximum_delay.is_zero() {
                std::thread::sleep(maximum_delay);
            }
            0
        }
        _ => unreachable!("caller filters AudioOut2 imports"),
    }
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
        let Some(entry_offset) =
            tls_dtv_entry_offset(module_id).and_then(|offset| usize::try_from(offset).ok())
        else {
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
            | "ytQULN-nhL4"
            | "1471ajPzxh0"
            | "iGjsr1WAtI0"
            | "sIlRvQqsN2Y"
            | "SFxTMOfuCkE"
            | "XhWHn6P5R7U"
            | "EgmLo6EWgso"
            | "2Tb92quprl0"
            | "0TyVk4MSLt0"
            | "g+PZd2hiacg"
            | "RXXqi4CtF8w"
            | "WKAXJ4XBPQ4"
            | "BmMjYxmew1w"
            | "27bAgiJmOh0"
            | "kDh-NfxgMtE"
            | "JGgj7Uvrl+A"
            | "mkx2fVhNMsg"
            | "2MOy+rUfuhQ"
            | "Op8TBGY5KHg"
            | "14bOACANTBo"
            | "188x57JYp0g"
            | "R1Jvn8bSCW8"
            | "Zxa0VhQVTsk"
            | "4czppHBiriw"
            | "BpFoboUJoZU"
            | "8mql9OcQnd4"
            | "IOnSvHzqu6A"
            | "7uhBFWRAS60"
            | "9lvj5DjHZiA"
            | "JTvBflhYazQ"
            | "PZku4ZrXJqg"
    )
}

#[cfg(test)]
fn dispatch_guest_sync_import(
    nid: &str,
    arguments: [u64; 6],
    mutex_attributes: Option<PthreadMutexAttrState>,
) -> u64 {
    dispatch_guest_sync_import_at_import(nid, arguments, mutex_attributes, None)
}

fn dispatch_guest_sync_import_at_import(
    nid: &str,
    arguments: [u64; 6],
    mutex_attributes: Option<PthreadMutexAttrState>,
    frame: Option<&ImportCallFrame>,
) -> u64 {
    match nid {
        "cmo1RIYva9o" | "ttHNfU+qDBU" => {
            guest_mutex_init(arguments[0], mutex_attributes.unwrap_or_default())
        }
        "2Of0f+3mhhE" | "ltCfaGr2JGE" => guest_mutex_destroy(arguments[0]),
        "9UK1vLZQft4" | "7H0iTOciTLo" => guest_mutex_lock(arguments[0], false),
        "upoVrzMHFeE" | "K-jXhbt2gn4" => guest_mutex_lock(arguments[0], true),
        "tn3VlD0hG60" | "2Z+PpY6CaJg" => guest_mutex_unlock(arguments[0]),
        "ytQULN-nhL4" => guest_rwlock_init(arguments[0]),
        "1471ajPzxh0" => guest_rwlock_destroy(arguments[0]),
        "iGjsr1WAtI0" => guest_rwlock_lock(arguments[0], false, false),
        "sIlRvQqsN2Y" => guest_rwlock_lock(arguments[0], true, false),
        "SFxTMOfuCkE" => guest_rwlock_lock(arguments[0], false, true),
        "XhWHn6P5R7U" => guest_rwlock_lock(arguments[0], true, true),
        "EgmLo6EWgso" => guest_rwlock_unlock(arguments[0]),
        "2Tb92quprl0" => guest_condition_init(arguments[0]),
        "0TyVk4MSLt0" => posix_pthread_result(guest_condition_init(arguments[0])),
        "g+PZd2hiacg" => guest_condition_destroy(arguments[0]),
        "RXXqi4CtF8w" => posix_pthread_result(guest_condition_destroy(arguments[0])),
        "WKAXJ4XBPQ4" => {
            guest_condition_wait_with_interrupt(arguments[0], arguments[1], None, || {
                if let Some(frame) = frame {
                    deliver_pending_guest_exception_at_import(frame);
                }
            })
        }
        "Op8TBGY5KHg" => posix_pthread_result(guest_condition_wait_with_interrupt(
            arguments[0],
            arguments[1],
            None,
            || {
                if let Some(frame) = frame {
                    deliver_pending_guest_exception_at_import(frame);
                }
            },
        )),
        "BmMjYxmew1w" => guest_condition_wait_with_interrupt(
            arguments[0],
            arguments[1],
            Some(u64::from(low_u32(arguments[2]))),
            || {
                if let Some(frame) = frame {
                    deliver_pending_guest_exception_at_import(frame);
                }
            },
        ),
        // The POSIX form takes an absolute deadline as a timespec; the console form takes a
        // relative microsecond count. Convert against the same realtime clock the deadline names.
        "27bAgiJmOh0" => posix_pthread_result(guest_condition_wait_with_interrupt(
            arguments[0],
            arguments[1],
            Some(relative_microseconds_until(arguments[2])),
            || {
                if let Some(frame) = frame {
                    deliver_pending_guest_exception_at_import(frame);
                }
            },
        )),
        "kDh-NfxgMtE" => guest_condition_notify(arguments[0], false),
        "2MOy+rUfuhQ" => posix_pthread_result(guest_condition_notify(arguments[0], false)),
        "JGgj7Uvrl+A" => guest_condition_notify(arguments[0], true),
        "mkx2fVhNMsg" => posix_pthread_result(guest_condition_notify(arguments[0], true)),
        "14bOACANTBo" => guest_pthread_once(arguments[0], arguments[1]),
        "188x57JYp0g" => guest_semaphore_create(arguments),
        "R1Jvn8bSCW8" => guest_semaphore_delete(arguments[0]),
        "Zxa0VhQVTsk" => guest_semaphore_wait(arguments),
        "4czppHBiriw" => guest_semaphore_signal(arguments),
        "BpFoboUJoZU" => guest_event_flag_create(arguments),
        "8mql9OcQnd4" => guest_event_flag_delete(arguments[0]),
        "IOnSvHzqu6A" => guest_event_flag_set(arguments[0], arguments[1]),
        "7uhBFWRAS60" => guest_event_flag_clear(arguments[0], arguments[1]),
        "9lvj5DjHZiA" => guest_event_flag_poll(arguments),
        "JTvBflhYazQ" => guest_event_flag_wait(arguments),
        "PZku4ZrXJqg" => guest_event_flag_cancel(arguments),
        _ => 0,
    }
}

/// Applies the caller's error convention to a thread-join result.
fn join_result(result: u64, posix: bool) -> u64 {
    if posix {
        posix_pthread_result(result)
    } else {
        result
    }
}

/// Converts an absolute `timespec` deadline into microseconds from now.
///
/// A deadline already in the past yields zero, which the wait treats as an immediate timeout
/// rather than an unbounded block.
fn relative_microseconds_until(deadline: u64) -> u64 {
    if deadline == 0 {
        return 0;
    }
    let seconds = i64::from_ne_bytes(read_guest_u64(deadline).to_ne_bytes());
    let nanoseconds = i64::from_ne_bytes(read_guest_u64(deadline + 8).to_ne_bytes());
    let deadline_microseconds = seconds
        .saturating_mul(1_000_000)
        .saturating_add(nanoseconds.div_euclid(1_000));
    let now_microseconds = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |elapsed| {
            i64::try_from(elapsed.as_micros()).unwrap_or(i64::MAX)
        });
    u64::try_from(deadline_microseconds.saturating_sub(now_microseconds)).unwrap_or(0)
}

fn posix_pthread_result(result: u64) -> u64 {
    let result32 = low_u32(result);
    if result32 & 0xffff_0000 == 0x8002_0000 {
        u64::from(result32 & 0xffff)
    } else {
        result
    }
}

fn guest_sync() -> &'static (Mutex<GuestSyncState>, Condvar) {
    GUEST_SYNC.get_or_init(|| (Mutex::new(GuestSyncState::default()), Condvar::new()))
}

fn guest_mutex_shards()
-> &'static [Mutex<HashMap<u64, Arc<GuestMutexEntry>>>; GUEST_MUTEX_SHARD_COUNT] {
    GUEST_MUTEX_SHARDS.get_or_init(|| std::array::from_fn(|_| Mutex::new(HashMap::new())))
}

fn guest_mutex_shard(address: u64) -> usize {
    usize::try_from(address >> 4).unwrap_or_default() & (GUEST_MUTEX_SHARD_COUNT - 1)
}

fn guest_mutex_entry(address: u64, create: bool) -> Option<Arc<GuestMutexEntry>> {
    let mut shard = guest_mutex_shards()[guest_mutex_shard(address)]
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    if let Some(entry) = shard.get(&address) {
        return Some(entry.clone());
    }
    create.then(|| {
        let entry = Arc::new(GuestMutexEntry {
            state: Mutex::new(GuestMutexState::default()),
            wake: Condvar::new(),
        });
        shard.insert(address, entry.clone());
        entry
    })
}

fn guest_pthread_once(control: u64, routine: u64) -> u64 {
    const NEVER_DONE: u32 = 0;
    const DONE: u32 = 1;
    const IN_PROGRESS: u32 = 2;
    const WAIT: u32 = 3;
    if control == 0 || routine == 0 {
        return sce_kernel_error(SCE_KERNEL_ERROR_INVALID_ARGUMENT);
    }
    let sync = guest_sync();
    let mut state = sync
        .0
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    loop {
        match read_guest_u32(control) {
            DONE => return 0,
            NEVER_DONE => {
                write_guest_u32(control, IN_PROGRESS);
                break;
            }
            IN_PROGRESS | WAIT => {
                write_guest_u32(control, WAIT);
                state = sync
                    .1
                    .wait(state)
                    .unwrap_or_else(std::sync::PoisonError::into_inner);
            }
            _ => return sce_kernel_error(SCE_KERNEL_ERROR_INVALID_ARGUMENT),
        }
    }
    drop(state);

    type OnceRoutine = unsafe extern "C" fn();
    // SAFETY: scePthreadOnce's ABI requires a live no-argument guest function pointer. No REmu
    // state lock is held while it runs, so initialization may call back through other imports.
    let routine: OnceRoutine = unsafe { std::mem::transmute(routine) };
    unsafe { routine() };

    let _state = sync
        .0
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    write_guest_u32(control, DONE);
    sync.1.notify_all();
    0
}

fn guest_mutex_init(address: u64, attributes: PthreadMutexAttrState) -> u64 {
    if address == 0 {
        return sce_kernel_error(SCE_KERNEL_ERROR_INVALID_ARGUMENT);
    }
    let mut shard = guest_mutex_shards()[guest_mutex_shard(address)]
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    if shard.contains_key(&address) {
        return sce_kernel_error(SCE_KERNEL_ERROR_INVALID_ARGUMENT);
    }
    shard.insert(
        address,
        Arc::new(GuestMutexEntry {
            state: Mutex::new(GuestMutexState {
                mutex_type: attributes.mutex_type,
                ..GuestMutexState::default()
            }),
            wake: Condvar::new(),
        }),
    );
    0
}

fn guest_mutex_destroy(address: u64) -> u64 {
    if address == 0 {
        return sce_kernel_error(SCE_KERNEL_ERROR_INVALID_ARGUMENT);
    }
    let mut shard = guest_mutex_shards()[guest_mutex_shard(address)]
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let Some(entry) = shard.get(&address).cloned() else {
        return sce_kernel_error(SCE_KERNEL_ERROR_NOT_FOUND);
    };
    let mutex = entry
        .state
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    if mutex.owner.is_some() || mutex.waiters != 0 {
        return u64::from(libc::EBUSY.cast_unsigned());
    }
    drop(mutex);
    shard.remove(&address);
    0
}

fn guest_mutex_lock(address: u64, try_only: bool) -> u64 {
    if address == 0 {
        return u64::MAX;
    }
    let Some(entry) = guest_mutex_entry(address, true) else {
        return u64::MAX;
    };
    let current = std::thread::current().id();
    let mut mutex = entry
        .state
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    loop {
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
                mutex.waiters = mutex.waiters.saturating_add(1);
                mutex = entry
                    .wake
                    .wait(mutex)
                    .unwrap_or_else(std::sync::PoisonError::into_inner);
                mutex.waiters = mutex.waiters.saturating_sub(1);
            }
        }
    }
}

fn guest_mutex_unlock(address: u64) -> u64 {
    let Some(entry) = guest_mutex_entry(address, false) else {
        return 0;
    };
    let current = std::thread::current().id();
    let mut mutex = entry
        .state
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    if mutex.owner.as_ref().is_some_and(|owner| *owner == current) {
        if mutex.recursion > 1 {
            mutex.recursion -= 1;
        } else {
            mutex.owner = None;
            mutex.recursion = 0;
            entry.wake.notify_one();
        }
    }
    0
}

fn guest_rwlock_init(address: u64) -> u64 {
    if address == 0 {
        return u64::from(libc::EINVAL.cast_unsigned());
    }
    let sync = guest_sync();
    let mut state = sync
        .0
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    if state.rwlocks.contains_key(&address) {
        return u64::from(libc::EBUSY.cast_unsigned());
    }
    state.rwlocks.insert(address, GuestRwlockState::default());
    0
}

fn guest_rwlock_destroy(address: u64) -> u64 {
    if address == 0 {
        return u64::from(libc::EINVAL.cast_unsigned());
    }
    let sync = guest_sync();
    let mut state = sync
        .0
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    match state.rwlocks.get(&address) {
        Some(lock)
            if lock.writer.is_some() || lock.reader_total != 0 || lock.waiting_writers != 0 =>
        {
            u64::from(libc::EBUSY.cast_unsigned())
        }
        Some(_) => {
            state.rwlocks.remove(&address);
            0
        }
        None => u64::from(libc::EINVAL.cast_unsigned()),
    }
}

fn guest_rwlock_lock(address: u64, write: bool, try_only: bool) -> u64 {
    if address == 0 {
        return u64::from(libc::EINVAL.cast_unsigned());
    }
    let sync = guest_sync();
    let current = std::thread::current().id();
    let mut state = sync
        .0
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    state.rwlocks.entry(address).or_default();

    if write {
        if state
            .rwlocks
            .get(&address)
            .is_some_and(|lock| lock.writer.as_ref() == Some(&current))
        {
            return u64::from(libc::EDEADLK.cast_unsigned());
        }
        let acquired = state
            .rwlocks
            .get(&address)
            .is_some_and(|lock| lock.writer.is_none() && lock.reader_total == 0);
        if acquired {
            state
                .rwlocks
                .get_mut(&address)
                .expect("rwlock was initialized")
                .writer = Some(current);
            return 0;
        }
        if try_only {
            return u64::from(libc::EBUSY.cast_unsigned());
        }
        state
            .rwlocks
            .get_mut(&address)
            .expect("rwlock was initialized")
            .waiting_writers += 1;
        loop {
            let acquired = state
                .rwlocks
                .get(&address)
                .is_some_and(|lock| lock.writer.is_none() && lock.reader_total == 0);
            if acquired {
                let lock = state
                    .rwlocks
                    .get_mut(&address)
                    .expect("a waiting writer prevents rwlock destruction");
                lock.waiting_writers -= 1;
                lock.writer = Some(current);
                return 0;
            }
            state = sync
                .1
                .wait(state)
                .unwrap_or_else(std::sync::PoisonError::into_inner);
        }
    }

    loop {
        let acquired = state
            .rwlocks
            .get(&address)
            .is_some_and(|lock| lock.writer.is_none() && lock.waiting_writers == 0);
        if acquired {
            let lock = state
                .rwlocks
                .get_mut(&address)
                .expect("rwlock was initialized");
            *lock.readers.entry(current).or_default() += 1;
            lock.reader_total += 1;
            return 0;
        }
        if try_only {
            return u64::from(libc::EBUSY.cast_unsigned());
        }
        state = sync
            .1
            .wait(state)
            .unwrap_or_else(std::sync::PoisonError::into_inner);
    }
}

fn guest_rwlock_unlock(address: u64) -> u64 {
    let sync = guest_sync();
    let current = std::thread::current().id();
    let mut state = sync
        .0
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let Some(lock) = state.rwlocks.get_mut(&address) else {
        return u64::from(libc::EINVAL.cast_unsigned());
    };
    if lock.writer.as_ref() == Some(&current) {
        lock.writer = None;
        sync.1.notify_all();
        return 0;
    }
    let Some(recursion) = lock.readers.get_mut(&current) else {
        return u64::from(libc::EPERM.cast_unsigned());
    };
    *recursion -= 1;
    lock.reader_total -= 1;
    if *recursion == 0 {
        lock.readers.remove(&current);
    }
    if lock.reader_total == 0 {
        sync.1.notify_all();
    }
    0
}

fn guest_condition_init(address: u64) -> u64 {
    if address == 0 {
        return sce_kernel_error(SCE_KERNEL_ERROR_INVALID_ARGUMENT);
    }
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
    if address == 0 {
        return sce_kernel_error(SCE_KERNEL_ERROR_INVALID_ARGUMENT);
    }
    let sync = guest_sync();
    let mut state = sync
        .0
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    match state.conditions.get(&address) {
        Some(condition) if !condition.is_idle() => sce_kernel_error(SCE_KERNEL_ERROR_BUSY),
        Some(_) => {
            state.conditions.remove(&address);
            0
        }
        None => sce_kernel_error(SCE_KERNEL_ERROR_INVALID_ARGUMENT),
    }
}

#[cfg(test)]
fn guest_condition_wait(condition: u64, mutex_address: u64, timeout_usec: Option<u64>) -> u64 {
    guest_condition_wait_with_interrupt(condition, mutex_address, timeout_usec, || {})
}

fn guest_condition_wait_with_interrupt<F>(
    condition: u64,
    mutex_address: u64,
    timeout_usec: Option<u64>,
    mut on_interrupt: F,
) -> u64
where
    F: FnMut(),
{
    if condition == 0 || mutex_address == 0 {
        return sce_kernel_error(SCE_KERNEL_ERROR_INVALID_ARGUMENT);
    }
    let sync = guest_sync();
    let current = std::thread::current().id();
    let Some(mutex_entry) = guest_mutex_entry(mutex_address, false) else {
        return sce_kernel_error(SCE_KERNEL_ERROR_OPERATION_NOT_PERMITTED);
    };
    let mut mutex = mutex_entry
        .state
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let released_recursion = if mutex.owner.as_ref().is_some_and(|owner| *owner == current) {
        mutex.recursion.max(1)
    } else {
        return sce_kernel_error(SCE_KERNEL_ERROR_OPERATION_NOT_PERMITTED);
    };
    let mut state = sync
        .0
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let Some(waiter) = state.register_condition_waiter(condition, mutex_address) else {
        return sce_kernel_error(SCE_KERNEL_ERROR_INVALID_ARGUMENT);
    };
    if trace_guest_sync() && trace_logarithmic_sequence(waiter, 8) {
        eprintln!(
            "[REMu][COND] wait ticket={waiter} thread={} condition={condition:#x} mutex={mutex_address:#x} recursion={released_recursion}",
            current_guest_thread_handle()
        );
    }
    mutex.owner = None;
    mutex.recursion = 0;
    mutex_entry.wake.notify_one();
    drop(mutex);
    let guest_thread = current_guest_thread_handle();

    let (timed_out, interrupted) = if let Some(timeout) = timeout_usec {
        let duration = Duration::from_micros(timeout);
        let (next_state, wait_result) = sync
            .1
            .wait_timeout_while(state, duration, |state| {
                !state.condition_waiter_released(waiter)
                    && !guest_exception_is_pending(guest_thread)
            })
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        state = next_state;
        let interrupted =
            !state.condition_waiter_released(waiter) && guest_exception_is_pending(guest_thread);
        (
            wait_result.timed_out() && !state.condition_waiter_released(waiter) && !interrupted,
            interrupted,
        )
    } else {
        state = sync
            .1
            .wait_while(state, |state| {
                !state.condition_waiter_released(waiter)
                    && !guest_exception_is_pending(guest_thread)
            })
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let interrupted =
            !state.condition_waiter_released(waiter) && guest_exception_is_pending(guest_thread);
        (false, interrupted)
    };
    state.finish_condition_wait(condition, waiter, timed_out || interrupted);
    drop(state);

    if interrupted {
        // A cooperative guest exception is analogous to a signal interrupting pthread_cond_wait.
        // Run the handler before trying to reacquire the guest mutex: a stop-the-world handler can
        // suspend the mutex owner, so reacquiring first creates a circular suspension deadlock.
        // Once the handler returns (the guest has resumed), normal cond-wait semantics still
        // require this thread to reacquire the mutex before returning to guest code.
        on_interrupt();
    }

    let mut mutex = mutex_entry
        .state
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    loop {
        if mutex.owner.is_none() {
            mutex.owner = Some(current);
            mutex.recursion = released_recursion;
            return if timed_out {
                sce_kernel_error(SCE_KERNEL_ERROR_TIMED_OUT)
            } else {
                0
            };
        }
        mutex.waiters = mutex.waiters.saturating_add(1);
        mutex = mutex_entry
            .wake
            .wait(mutex)
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        mutex.waiters = mutex.waiters.saturating_sub(1);
    }
}

fn guest_condition_notify(condition: u64, broadcast: bool) -> u64 {
    if condition == 0 {
        return sce_kernel_error(SCE_KERNEL_ERROR_INVALID_ARGUMENT);
    }
    let sync = guest_sync();
    let mut state = sync
        .0
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    state.condition_notify_count = state.condition_notify_count.saturating_add(1);
    let notify_sequence = state.condition_notify_count;
    let guest_condition = state.conditions.entry(condition).or_default();
    let pending_before = guest_condition.waiters.len();
    let selected = if broadcast {
        guest_condition.release_all()
    } else {
        guest_condition.release_one().into_iter().collect()
    };
    for waiter in &selected {
        state
            .condition_waiters
            .get_mut(waiter)
            .expect("pending condition waiter exists")
            .released = true;
    }
    if trace_guest_sync() && trace_logarithmic_sequence(notify_sequence, 8) {
        eprintln!(
            "[REMu][COND] notify sequence={notify_sequence} condition={condition:#x} broadcast={broadcast} pending={pending_before} selected={selected:?}"
        );
    }
    if !selected.is_empty() {
        // One host Condvar coordinates all guest synchronization objects. Wake all host waiters so
        // the selected condition ticket cannot be consumed by an unrelated host waiter; only the
        // released guest ticket is allowed to proceed.
        sync.1.notify_all();
    }
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
    let guest_thread = current_guest_thread_handle();
    let mut state = sync
        .0
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    state.semaphore_wait_count = state.semaphore_wait_count.saturating_add(1);
    let wait_sequence = state.semaphore_wait_count;
    let trace_call = trace_guest_sync() && trace_logarithmic_sequence(wait_sequence, 8);
    let mut logged_block = false;
    loop {
        let Some(semaphore) = state.semaphores.get_mut(&handle) else {
            return u64::MAX;
        };
        if semaphore.count >= needed {
            semaphore.count -= needed;
            if trace_call {
                eprintln!(
                    "[REMu][SEMA] acquire sequence={wait_sequence} handle={handle} needed={needed} remaining={}",
                    semaphore.count
                );
            }
            return 0;
        }
        if guest_exception_is_pending(guest_thread) {
            return sce_kernel_error(SCE_KERNEL_ERROR_INTERRUPTED);
        }
        if trace_call && !logged_block {
            eprintln!(
                "[REMu][SEMA] wait sequence={wait_sequence} handle={handle} needed={needed} available={}",
                semaphore.count
            );
            logged_block = true;
        }
        state = sync
            .1
            .wait(state)
            .unwrap_or_else(std::sync::PoisonError::into_inner);
    }
}

fn guest_semaphore_delete(handle: u64) -> u64 {
    let sync = guest_sync();
    let mut state = sync
        .0
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    if state.semaphores.remove(&low_u32(handle)).is_none() {
        return sce_kernel_error(SCE_KERNEL_ERROR_NOT_FOUND);
    }
    sync.1.notify_all();
    0
}

fn guest_semaphore_signal(arguments: [u64; 6]) -> u64 {
    let handle = low_u32(arguments[0]);
    let amount = low_i32(arguments[1]);
    let sync = guest_sync();
    let mut state = sync
        .0
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    state.semaphore_signal_count = state.semaphore_signal_count.saturating_add(1);
    let signal_sequence = state.semaphore_signal_count;
    let Some(semaphore) = state.semaphores.get_mut(&handle) else {
        return u64::MAX;
    };
    semaphore.count = semaphore
        .count
        .saturating_add(amount)
        .min(semaphore.maximum);
    if trace_guest_sync() && trace_logarithmic_sequence(signal_sequence, 8) {
        eprintln!(
            "[REMu][SEMA] signal sequence={signal_sequence} handle={handle} amount={amount} available={}",
            semaphore.count
        );
    }
    sync.1.notify_all();
    0
}

const EVENT_FLAG_WAIT_AND: u32 = 0x01;
const EVENT_FLAG_WAIT_OR: u32 = 0x02;
const EVENT_FLAG_CLEAR_ALL: u32 = 0x10;
const EVENT_FLAG_CLEAR_PATTERN: u32 = 0x20;

fn event_flag_attributes_valid(attributes: u32) -> bool {
    matches!(attributes & 0x0f, 0 | 0x01 | 0x02)
        && matches!(attributes & 0xf0, 0 | 0x10 | 0x20)
        && attributes & !0x33 == 0
}

fn event_flag_wait_mode_valid(wait_mode: u32) -> bool {
    matches!(wait_mode & 0x0f, EVENT_FLAG_WAIT_AND | EVENT_FLAG_WAIT_OR)
        && matches!(
            wait_mode & 0xf0,
            0 | EVENT_FLAG_CLEAR_ALL | EVENT_FLAG_CLEAR_PATTERN
        )
        && wait_mode & !0x33 == 0
}

fn event_flag_satisfied(bits: u64, pattern: u64, wait_mode: u32) -> bool {
    if wait_mode & 0x0f == EVENT_FLAG_WAIT_AND {
        bits & pattern == pattern
    } else {
        bits & pattern != 0
    }
}

fn event_flag_apply_clear(bits: &mut u64, pattern: u64, wait_mode: u32) {
    match wait_mode & 0xf0 {
        EVENT_FLAG_CLEAR_ALL => *bits = 0,
        EVENT_FLAG_CLEAR_PATTERN => *bits &= !pattern,
        _ => {}
    }
}

fn guest_event_flag_create(arguments: [u64; 6]) -> u64 {
    let output = arguments[0];
    let Some(name) = read_guest_cstring(arguments[1]) else {
        return sce_kernel_error(SCE_KERNEL_ERROR_MEMORY_FAULT);
    };
    let attributes = low_u32(arguments[2]);
    if output == 0
        || arguments[1] == 0
        || arguments[4] != 0
        || name.len() > 31
        || !event_flag_attributes_valid(attributes)
    {
        return sce_kernel_error(SCE_KERNEL_ERROR_INVALID_ARGUMENT);
    }
    let sync = guest_sync();
    let mut state = sync
        .0
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let handle = loop {
        state.next_event_flag_handle = state.next_event_flag_handle.wrapping_add(1).max(1);
        if !state
            .event_flags
            .contains_key(&state.next_event_flag_handle)
        {
            break state.next_event_flag_handle;
        }
    };
    state.event_flags.insert(
        handle,
        GuestEventFlagState {
            name: name.clone(),
            attributes,
            bits: arguments[3],
            waiters: 0,
            cancellation_generation: 0,
        },
    );
    write_guest_u64(output, handle);
    if trace_guest_sync() {
        eprintln!(
            "[REMu][EVENT_FLAG] create handle={handle} name={name:?} attr={attributes:#x} bits={:#x}",
            arguments[3]
        );
    }
    0
}

fn guest_event_flag_delete(handle: u64) -> u64 {
    let sync = guest_sync();
    let mut state = sync
        .0
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let Some(flag) = state.event_flags.remove(&handle) else {
        return sce_kernel_error(SCE_KERNEL_ERROR_NOT_FOUND);
    };
    if trace_guest_sync() {
        eprintln!(
            "[REMu][EVENT_FLAG] delete handle={handle} name={:?} attr={:#x} waiters={}",
            flag.name, flag.attributes, flag.waiters
        );
    }
    sync.1.notify_all();
    0
}

fn guest_event_flag_set(handle: u64, pattern: u64) -> u64 {
    let sync = guest_sync();
    let mut state = sync
        .0
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let Some(flag) = state.event_flags.get_mut(&handle) else {
        return sce_kernel_error(SCE_KERNEL_ERROR_NOT_FOUND);
    };
    flag.bits |= pattern;
    sync.1.notify_all();
    0
}

fn guest_event_flag_clear(handle: u64, mask: u64) -> u64 {
    let sync = guest_sync();
    let mut state = sync
        .0
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let Some(flag) = state.event_flags.get_mut(&handle) else {
        return sce_kernel_error(SCE_KERNEL_ERROR_NOT_FOUND);
    };
    flag.bits &= mask;
    0
}

fn guest_event_flag_poll(arguments: [u64; 6]) -> u64 {
    let pattern = arguments[1];
    let wait_mode = low_u32(arguments[2]);
    if pattern == 0 || !event_flag_wait_mode_valid(wait_mode) {
        return sce_kernel_error(SCE_KERNEL_ERROR_INVALID_ARGUMENT);
    }
    let sync = guest_sync();
    let mut state = sync
        .0
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let Some(flag) = state.event_flags.get_mut(&arguments[0]) else {
        return sce_kernel_error(SCE_KERNEL_ERROR_NOT_FOUND);
    };
    if arguments[3] != 0 {
        write_guest_u64(arguments[3], flag.bits);
    }
    if !event_flag_satisfied(flag.bits, pattern, wait_mode) {
        return sce_kernel_error(SCE_KERNEL_ERROR_BUSY);
    }
    event_flag_apply_clear(&mut flag.bits, pattern, wait_mode);
    0
}

fn guest_event_flag_wait(arguments: [u64; 6]) -> u64 {
    let handle = arguments[0];
    let pattern = arguments[1];
    let wait_mode = low_u32(arguments[2]);
    if pattern == 0 || !event_flag_wait_mode_valid(wait_mode) {
        return sce_kernel_error(SCE_KERNEL_ERROR_INVALID_ARGUMENT);
    }
    let timeout = (arguments[4] != 0).then(|| u64::from(read_guest_u32(arguments[4])));
    let deadline = timeout
        .and_then(|microseconds| Instant::now().checked_add(Duration::from_micros(microseconds)));
    let sync = guest_sync();
    let guest_thread = current_guest_thread_handle();
    let mut state = sync
        .0
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let Some(flag) = state.event_flags.get_mut(&handle) else {
        return sce_kernel_error(SCE_KERNEL_ERROR_NOT_FOUND);
    };
    let cancellation_generation = flag.cancellation_generation;
    if event_flag_satisfied(flag.bits, pattern, wait_mode) {
        if arguments[3] != 0 {
            write_guest_u64(arguments[3], flag.bits);
        }
        event_flag_apply_clear(&mut flag.bits, pattern, wait_mode);
        return 0;
    }
    flag.waiters = flag.waiters.saturating_add(1);
    loop {
        let Some(flag) = state.event_flags.get_mut(&handle) else {
            return sce_kernel_error(SCE_KERNEL_ERROR_NOT_FOUND);
        };
        if guest_exception_is_pending(guest_thread) {
            if arguments[3] != 0 {
                write_guest_u64(arguments[3], flag.bits);
            }
            flag.waiters = flag.waiters.saturating_sub(1);
            return sce_kernel_error(SCE_KERNEL_ERROR_INTERRUPTED);
        }
        if flag.cancellation_generation != cancellation_generation {
            flag.waiters = flag.waiters.saturating_sub(1);
            return sce_kernel_error(SCE_KERNEL_ERROR_INTERRUPTED);
        }
        if event_flag_satisfied(flag.bits, pattern, wait_mode) {
            if arguments[3] != 0 {
                write_guest_u64(arguments[3], flag.bits);
            }
            event_flag_apply_clear(&mut flag.bits, pattern, wait_mode);
            flag.waiters = flag.waiters.saturating_sub(1);
            return 0;
        }
        if let Some(deadline) = deadline {
            let Some(remaining) = deadline.checked_duration_since(Instant::now()) else {
                if arguments[3] != 0 {
                    write_guest_u64(arguments[3], flag.bits);
                }
                if arguments[4] != 0 {
                    write_guest_u32(arguments[4], 0);
                }
                flag.waiters = flag.waiters.saturating_sub(1);
                return sce_kernel_error(SCE_KERNEL_ERROR_TIMED_OUT);
            };
            let (next_state, wait_result) = sync
                .1
                .wait_timeout(state, remaining)
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            state = next_state;
            if wait_result.timed_out() {
                continue;
            }
        } else {
            state = sync
                .1
                .wait(state)
                .unwrap_or_else(std::sync::PoisonError::into_inner);
        }
    }
}

fn guest_event_flag_cancel(arguments: [u64; 6]) -> u64 {
    let sync = guest_sync();
    let mut state = sync
        .0
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let Some(flag) = state.event_flags.get_mut(&arguments[0]) else {
        return sce_kernel_error(SCE_KERNEL_ERROR_NOT_FOUND);
    };
    if arguments[2] != 0 {
        write_guest_u32(arguments[2], flag.waiters);
    }
    flag.bits = arguments[1];
    flag.cancellation_generation = flag.cancellation_generation.wrapping_add(1);
    sync.1.notify_all();
    0
}

fn low_u32(value: u64) -> u32 {
    let bytes = value.to_le_bytes();
    u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]])
}

const fn trace_logarithmic_sequence(sequence: u64, initial: u64) -> bool {
    sequence <= initial || sequence.is_power_of_two()
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
        "1-LFLmRFxxM" => guest_kernel_mkdir(arguments, state),
        "naInUjYt3so" => guest_kernel_rmdir(arguments, state),
        "eV9wAD2riIA" => guest_kernel_stat(arguments, state),
        "Cg4srZ6TKbU" => guest_kernel_read(arguments),
        "UK2Tl2DWUns" => guest_kernel_close(arguments),
        "oib76F-12fk" => guest_kernel_lseek(arguments),
        "kBwCPsYX-m4" => guest_kernel_fstat(arguments),
        "VW3TVZiM4-E" => guest_kernel_ftruncate(arguments),
        _ => return None,
    })
}

/// File-descriptor operations are independent of the process service graph. FMOD performs reads
/// and seeks on its streaming thread while the graphics thread can spend tens of milliseconds
/// translating one AGC submission. Keeping these calls behind `PROBE_DISPATCH` starved the audio
/// producer even though it touched a completely unrelated file.
fn dispatch_kernel_fd_import(nid: &str, arguments: [u64; 6]) -> Option<u64> {
    Some(match nid {
        // Raw libKernel error convention.
        "Cg4srZ6TKbU" => guest_kernel_read(arguments),
        "UK2Tl2DWUns" => guest_kernel_close(arguments),
        "oib76F-12fk" => guest_kernel_lseek(arguments),
        "kBwCPsYX-m4" => guest_kernel_fstat(arguments),
        "VW3TVZiM4-E" => guest_kernel_ftruncate(arguments),
        // POSIX wrappers translate kernel errors to -1 and guest errno.
        "DRuBt2pvICk" | "AqBioC2vF3I" => guest_posix_file_result(guest_kernel_read(arguments)),
        "NNtFaKJbPt0" | "bY-PO6JhzhQ" => guest_posix_file_result(guest_kernel_close(arguments)),
        "FxVZqBAA7ks" | "FN4gaPmuFV8" => guest_posix_file_result(guest_kernel_write(arguments)),
        "Oy6IpwgtYOk" => guest_posix_file_result(guest_kernel_lseek(arguments)),
        "mqQMh1zPPT8" => guest_posix_file_result(guest_kernel_fstat(arguments)),
        "ih4CD9-gghM" => guest_posix_file_result(guest_kernel_ftruncate(arguments)),
        "4wSze92BhLI" => guest_kernel_write(arguments),
        _ => return None,
    })
}

fn guest_kernel_rmdir(arguments: [u64; 6], state: &ProbeDispatchState) -> u64 {
    let Some(guest_path) = read_guest_cstring(arguments[0]) else {
        return sce_kernel_error(SCE_KERNEL_ERROR_MEMORY_FAULT);
    };
    let Some((host_path, writable)) = resolve_probe_guest_path(state, &guest_path) else {
        return sce_kernel_error(SCE_KERNEL_ERROR_NOT_FOUND);
    };
    if !writable {
        return sce_kernel_error(SCE_KERNEL_ERROR_PERMISSION_DENIED);
    }
    if !host_path.exists() {
        return sce_kernel_error(SCE_KERNEL_ERROR_NOT_FOUND);
    }
    if !host_path.is_dir() {
        return sce_kernel_error(SCE_KERNEL_ERROR_NOT_DIRECTORY);
    }
    // Only an empty directory may be removed; the guest deletes contents explicitly first.
    match std::fs::remove_dir(&host_path) {
        Ok(()) => 0,
        Err(error) => sce_kernel_io_error(&error),
    }
}

fn guest_kernel_mkdir(arguments: [u64; 6], state: &ProbeDispatchState) -> u64 {
    let Some(guest_path) = read_guest_cstring(arguments[0]) else {
        return sce_kernel_error(SCE_KERNEL_ERROR_MEMORY_FAULT);
    };
    let Some((host_path, writable)) = resolve_probe_guest_path(state, &guest_path) else {
        return sce_kernel_error(SCE_KERNEL_ERROR_NOT_FOUND);
    };
    if !writable {
        return sce_kernel_error(SCE_KERNEL_ERROR_PERMISSION_DENIED);
    }
    if host_path.exists() {
        return sce_kernel_error(SCE_KERNEL_ERROR_ALREADY_EXISTS);
    }
    let Some(parent) = host_path.parent() else {
        return sce_kernel_error(SCE_KERNEL_ERROR_NOT_FOUND);
    };
    if !parent.is_dir() {
        return sce_kernel_error(SCE_KERNEL_ERROR_NOT_FOUND);
    }
    match std::fs::create_dir(&host_path) {
        Ok(()) => 0,
        Err(error) => sce_kernel_io_error(&error),
    }
}

#[cfg(target_arch = "x86_64")]
type GuestModuleInitializer = unsafe extern "C" fn(u64, u64, u64) -> i32;

#[cfg(target_arch = "x86_64")]
fn invoke_guest_module_initializers(
    initializers: &[GuestAddress],
    argument_size: u64,
    argument: u64,
) -> i32 {
    let mut result = 0;
    for initializer in initializers {
        eprintln!(
            "[REMu][MODULE] initializing runtime at {initializer} args={argument_size:#x}/{argument:#x}"
        );
        // SAFETY: The module linker relocated this DT_INIT address in executable guest memory.
        let initializer: GuestModuleInitializer = unsafe { std::mem::transmute(initializer.get()) };
        // SAFETY: Gen5 module initializers use the SysV module-entry convention. The first two
        // values are sceKernelLoadStartModule's caller-owned argument block; a null third argument
        // asks the CRT wrapper to invoke the module's linked start function after its constructors.
        result = unsafe { initializer(argument_size, argument, 0) };
    }
    result
}

#[cfg(not(target_arch = "x86_64"))]
fn invoke_guest_module_initializers(
    _initializers: &[GuestAddress],
    _argument_size: u64,
    _argument: u64,
) -> i32 {
    0
}

fn kernel_load_start_module(arguments: [u64; 6]) -> u64 {
    if arguments[5] != 0 {
        write_guest_u32(arguments[5], 0);
    }
    let Some(requested_path) = read_guest_cstring(arguments[0]) else {
        let error = sce_kernel_error(SCE_KERNEL_ERROR_MEMORY_FAULT);
        if arguments[5] != 0 {
            write_guest_u32(arguments[5], low_u32(error));
        }
        return error;
    };
    let requested_name = Path::new(&requested_path)
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or(&requested_path);
    let dispatch = PROBE_DISPATCH.get_or_init(|| RwLock::new(ProbeDispatchState::default()));
    let mut state = dispatch
        .write()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let module = state.bundled_modules.iter_mut().find(|module| {
        module.name.eq_ignore_ascii_case(&requested_path)
            || Path::new(&module.name)
                .file_name()
                .and_then(|name| name.to_str())
                .is_some_and(|name| name.eq_ignore_ascii_case(requested_name))
    });
    let Some(module) = module else {
        let error = sce_kernel_error(SCE_KERNEL_ERROR_NOT_FOUND);
        if arguments[5] != 0 {
            write_guest_u32(arguments[5], low_u32(error));
        }
        eprintln!("[REMu][MODULE] load-start could not find {requested_path}");
        return error;
    };
    let handle = module.handle;
    let module_name = module.name.clone();
    let initializers = if module.started {
        Vec::new()
    } else {
        module.started = true;
        module.initializers.clone()
    };
    drop(state);
    let start_result = invoke_guest_module_initializers(&initializers, arguments[1], arguments[2]);
    if arguments[5] != 0 {
        write_guest_u32(arguments[5], start_result.cast_unsigned());
    }
    eprintln!(
        "[REMu][MODULE] load-start {requested_path} -> handle {} ({}) result={start_result:#x}",
        handle, module_name,
    );
    u64::from(handle)
}

fn sce_symbol_nid(symbol: &str) -> String {
    const SALT: [u8; 16] = [
        0x51, 0x8d, 0x64, 0xa6, 0x35, 0xde, 0xd8, 0xc1, 0xe6, 0xb0, 0x39, 0xb1, 0xc3, 0xe5, 0x52,
        0x30,
    ];
    const CODES: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+-";
    let mut hasher = Sha1::new();
    hasher.update(symbol.as_bytes());
    hasher.update(SALT);
    let hash = hasher.finalize();
    let digest = u64::from_le_bytes(hash[..8].try_into().expect("SHA-1 prefix has eight bytes"));
    let mut nid = String::with_capacity(11);
    for index in 0..10 {
        let shift = 58 - index * 6;
        nid.push(char::from(CODES[((digest >> shift) & 0x3f) as usize]));
    }
    nid.push(char::from(CODES[((digest & 0x0f) * 4) as usize]));
    nid
}

extern "C" fn trace_fmod_core_system_initialize(
    system: u64,
    maximum_channels: i32,
    core_flags: u32,
    extra_driver_data: u64,
) -> i32 {
    let target = FMOD_CORE_SYSTEM_INITIALIZE_TARGET.load(Ordering::Acquire);
    if target == 0 {
        return -1;
    }
    eprintln!(
        "[REMu][FMOD] Core_System_Init system={system:#x} channels={maximum_channels} flags={core_flags:#x} extra={extra_driver_data:#x}"
    );
    type Initialize = unsafe extern "C" fn(u64, i32, u32, u64) -> i32;
    // SAFETY: the linker records the mapped executable guest export with this exact FMOD C ABI
    // before redirecting the Studio module's import relocation to this tracing wrapper.
    let initialize: Initialize = unsafe { std::mem::transmute(target) };
    // SAFETY: the wrapper forwards the caller's unmodified C ABI arguments to the recorded export.
    let result = unsafe { initialize(system, maximum_channels, core_flags, extra_driver_data) };
    eprintln!("[REMu][FMOD] Core_System_Init result={result} ({result:#x})");
    result
}

extern "C" fn trace_fmod_studio_system_initialize(
    system: u64,
    maximum_channels: i32,
    studio_flags: u32,
    core_flags: u32,
    extra_driver_data: u64,
) -> i32 {
    let target = FMOD_STUDIO_SYSTEM_INITIALIZE_TARGET.load(Ordering::Acquire);
    if target == 0 {
        return -1;
    }
    eprintln!(
        "[REMu][FMOD] Studio_System_Initialize system={system:#x} channels={maximum_channels} studio_flags={studio_flags:#x} core_flags={core_flags:#x} extra={extra_driver_data:#x}"
    );
    type Initialize = unsafe extern "C" fn(u64, i32, u32, u32, u64) -> i32;
    // SAFETY: dlsym records the mapped executable guest export with this exact FMOD C ABI before
    // exposing the tracing wrapper to the caller. The guest image stays mapped for the process run.
    let initialize: Initialize = unsafe { std::mem::transmute(target) };
    // SAFETY: the wrapper forwards the caller's unmodified C ABI arguments to the recorded export.
    let result = unsafe {
        initialize(
            system,
            maximum_channels,
            studio_flags,
            core_flags,
            extra_driver_data,
        )
    };
    eprintln!("[REMu][FMOD] Studio_System_Initialize result={result} ({result:#x})");
    result
}

fn kernel_dlsym(arguments: [u64; 6], state: &ProbeDispatchState) -> u64 {
    let Some(symbol) = read_guest_cstring(arguments[1]) else {
        return u64::MAX;
    };
    let hle_target = match symbol.as_str() {
        "scriptingGetMem" => scripting_get_mem as *const () as usize as u64,
        "scriptingFreeMem" => libc::free as *const () as usize as u64,
        "scriptingRealloc" => libc::realloc as *const () as usize as u64,
        "scriptingCalloc" => libc::calloc as *const () as usize as u64,
        _ => 0,
    };
    let handle = low_u32(arguments[0]);
    let nid = sce_symbol_nid(&symbol);
    let guest_target = state
        .bundled_modules
        .iter()
        .find(|module| module.handle == handle)
        .and_then(|module| module.exports_by_nid.get(&nid))
        .copied()
        .unwrap_or_default();
    let mut target = if hle_target != 0 {
        hle_target
    } else {
        guest_target
    };
    if symbol == "FMOD_Studio_System_Initialize"
        && target != 0
        && std::env::var_os("REMU_TRACE_FMOD").is_some_and(|value| value == "1")
    {
        FMOD_STUDIO_SYSTEM_INITIALIZE_TARGET.store(
            usize::try_from(target).expect("guest function address fits the native pointer width"),
            Ordering::Release,
        );
        target = trace_fmod_studio_system_initialize as *const () as usize as u64;
    }
    if arguments[2] == 0 || target == 0 {
        if arguments[2] != 0 {
            write_guest_u64(arguments[2], 0);
        }
        eprintln!(
            "[REMu][MODULE] dlsym unresolved handle={} symbol={symbol:?} nid={nid}",
            arguments[0],
        );
        return u64::MAX;
    }
    write_guest_u64(arguments[2], target);
    eprintln!(
        "[REMu][MODULE] dlsym handle={} symbol={symbol:?} nid={nid} -> {target:#x}",
        arguments[0],
    );
    0
}

fn kernel_module_info_for_unwind(arguments: [u64; 6], state: &ProbeDispatchState) -> u64 {
    let queried_address = arguments[0];
    let flags = low_i32(arguments[1]);
    let output = arguments[2];
    if output == 0 || !(0..3).contains(&flags) {
        return sce_kernel_error(SCE_KERNEL_ERROR_INVALID_ARGUMENT);
    }
    if read_guest_u64(output) < 0x130 {
        return sce_kernel_error(SCE_KERNEL_ERROR_INVALID_ARGUMENT);
    }
    let Some(module) = state
        .unwind_modules
        .iter()
        .find(|module| (module.base..module.end).contains(&queried_address))
    else {
        return sce_kernel_error(SCE_KERNEL_ERROR_NOT_FOUND);
    };
    guest_memset(output, 0, 0x130);
    write_guest_u64(output, 0x130);
    let name = module.name.as_bytes();
    let name_length = name.len().min(0xff);
    write_guest_bytes(output + 8, &name[..name_length]);
    write_guest_u64(output + 0x108, module.eh_frame_header);
    write_guest_u64(output + 0x110, module.eh_frame);
    write_guest_u64(output + 0x118, module.eh_frame_size);
    write_guest_u64(output + 0x120, module.base);
    write_guest_u64(output + 0x128, module.end - module.base);
    0
}

extern "C" fn scripting_get_mem(alignment: u64, size: u64) -> u64 {
    guest_aligned_alloc(alignment, size)
}

fn kernel_install_exception_handler(arguments: [u64; 6], state: &mut ProbeDispatchState) -> u64 {
    let exception = low_i32(arguments[0]);
    if !matches!(exception, 1 | 4 | 8 | 10 | 11 | 30) || arguments[1] == 0 {
        return sce_kernel_error(SCE_KERNEL_ERROR_INVALID_ARGUMENT);
    }
    if state
        .installed_exception_handlers
        .insert(exception, arguments[1])
        .is_some()
    {
        return sce_kernel_error(SCE_KERNEL_ERROR_ALREADY_EXISTS);
    }
    eprintln!(
        "[REMu][KERNEL] exception handler installed type={exception} address={:#x}",
        arguments[1]
    );
    0
}

fn kernel_remove_exception_handler(exception: u64, state: &mut ProbeDispatchState) -> u64 {
    let exception = low_i32(exception);
    if !matches!(exception, 1 | 4 | 8 | 10 | 11 | 30) {
        return sce_kernel_error(SCE_KERNEL_ERROR_INVALID_ARGUMENT);
    }
    state.installed_exception_handlers.remove(&exception);
    0
}

fn kernel_raise_exception(arguments: [u64; 6], state: &ProbeDispatchState) -> u64 {
    let exception = low_i32(arguments[1]);
    if arguments[0] == 0 || !(0..128).contains(&exception) {
        return sce_kernel_error(SCE_KERNEL_ERROR_INVALID_ARGUMENT);
    }
    if let Some(handler) = state.installed_exception_handlers.get(&exception) {
        let Some(target_thread) = state.pthread_threads.get(&arguments[0]) else {
            eprintln!(
                "[REMu][KERNEL] exception target is not a live pthread: {:#x}",
                arguments[0]
            );
            return sce_kernel_error(SCE_KERNEL_ERROR_BUSY);
        };
        let source = current_guest_thread_handle();
        guest_exception_delivery()
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .pending
            .entry(arguments[0])
            .or_insert(PendingGuestException {
                handler: *handler,
                exception,
            });
        GUEST_EXCEPTION_PENDING.store(true, Ordering::Release);
        guest_sync().1.notify_all();
        eprintln!(
            "[REMu][KERNEL] queued cooperative exception source={source:#x} target={:#x} target_name={:?} type={exception} handler={handler:#x}",
            arguments[0], target_thread.name
        );
    }
    0
}

fn process_clock_nanos() -> u64 {
    u64::try_from(
        PROCESS_CLOCK_EPOCH
            .get_or_init(Instant::now)
            .elapsed()
            .as_nanos(),
    )
    .unwrap_or(u64::MAX)
}

fn reset_kernel_process_time() {
    PROCESS_START_NANOS.store(process_clock_nanos().max(1), Ordering::Release);
}

fn kernel_process_time(counter: bool) -> u64 {
    let now = process_clock_nanos();
    let mut start = PROCESS_START_NANOS.load(Ordering::Acquire);
    if start == 0 {
        // Unit-level dispatch tests can call this export without configuring a complete image.
        // Let the first such caller establish the same process-relative epoch without a mutex.
        match PROCESS_START_NANOS.compare_exchange(
            0,
            now.max(1),
            Ordering::AcqRel,
            Ordering::Acquire,
        ) {
            Ok(_) => start = now,
            Err(installed) => start = installed,
        }
    }
    let elapsed_nanos = now.saturating_sub(start);
    if counter {
        elapsed_nanos
    } else {
        elapsed_nanos / 1_000
    }
}

const SCE_AJM_ERROR_INVALID_PARAMETER: u32 = 0x8093_0005;
const SCE_AJM_ERROR_INVALID_CONTEXT: u32 = 0x8093_0002;
const SCE_AJM_ERROR_INVALID_INSTANCE: u32 = 0x8093_0003;
const SCE_AJM_ERROR_CODEC_ALREADY_REGISTERED: u32 = 0x8093_0009;
const SCE_AJM_ERROR_CODEC_NOT_REGISTERED: u32 = 0x8093_000a;

fn ajm_batch_initialize(arguments: [u64; 6]) -> u64 {
    let batch = arguments[0];
    let size = arguments[1];
    if batch == 0 || !(0x40..=0x10_000).contains(&size) {
        return sce_kernel_error(SCE_AJM_ERROR_INVALID_PARAMETER);
    }
    guest_memset(batch, 0, size);
    0
}

fn guest_zeroed_constructor(address: u64, size: u64) -> u64 {
    if address != 0 {
        guest_memset(address, 0, size);
    }
    address
}

fn ajm_initialize(arguments: [u64; 6], state: &mut ProbeDispatchState) -> u64 {
    let reserved = arguments[0];
    let output = arguments[1];
    if reserved != 0 || output == 0 {
        return sce_kernel_error(SCE_AJM_ERROR_INVALID_PARAMETER);
    }

    let context = state.next_ajm_context.max(1);
    state.next_ajm_context = context.saturating_add(1);
    state.ajm_contexts.insert(
        context,
        AjmContextState {
            next_instance: 1,
            ..AjmContextState::default()
        },
    );
    write_guest_u32(output, context);
    eprintln!("[REMu][AJM] initialized context={context}");
    0
}

fn ajm_module_register(arguments: [u64; 6], state: &mut ProbeDispatchState) -> u64 {
    let context = low_u32(arguments[0]);
    let codec = low_u32(arguments[1]);
    if arguments[2] != 0 {
        return sce_kernel_error(SCE_AJM_ERROR_INVALID_PARAMETER);
    }
    let Some(context_state) = state.ajm_contexts.get_mut(&context) else {
        return sce_kernel_error(SCE_AJM_ERROR_INVALID_CONTEXT);
    };
    if !context_state.registered_codecs.insert(codec) {
        return sce_kernel_error(SCE_AJM_ERROR_CODEC_ALREADY_REGISTERED);
    }
    eprintln!("[REMu][AJM] registered codec={codec} context={context}");
    0
}

fn ajm_module_unregister(arguments: [u64; 6], state: &mut ProbeDispatchState) -> u64 {
    let context = low_u32(arguments[0]);
    let codec = low_u32(arguments[1]);
    let Some(context_state) = state.ajm_contexts.get_mut(&context) else {
        return sce_kernel_error(SCE_AJM_ERROR_INVALID_CONTEXT);
    };
    context_state.registered_codecs.remove(&codec);
    eprintln!("[REMu][AJM] unregistered codec={codec} context={context}");
    0
}

fn ajm_instance_create(arguments: [u64; 6], state: &mut ProbeDispatchState) -> u64 {
    let context = low_u32(arguments[0]);
    let codec = low_u32(arguments[1]);
    let output = arguments[3];
    if output == 0 {
        return sce_kernel_error(SCE_AJM_ERROR_INVALID_PARAMETER);
    }
    let Some(context_state) = state.ajm_contexts.get_mut(&context) else {
        return sce_kernel_error(SCE_AJM_ERROR_INVALID_CONTEXT);
    };
    if !context_state.registered_codecs.contains(&codec) {
        return sce_kernel_error(SCE_AJM_ERROR_CODEC_NOT_REGISTERED);
    }
    let index = context_state.next_instance.max(1) & 0x3fff;
    context_state.next_instance = index.saturating_add(1);
    let instance = index | (codec << 14);
    context_state.instances.insert(instance);
    write_guest_u32(output, instance);
    eprintln!(
        "[REMu][AJM] created instance={instance:#x} codec={codec} context={context} flags={:#x}",
        arguments[2]
    );
    0
}

fn ajm_instance_destroy(arguments: [u64; 6], state: &mut ProbeDispatchState) -> u64 {
    let context = low_u32(arguments[0]);
    let instance = low_u32(arguments[1]);
    let Some(context_state) = state.ajm_contexts.get_mut(&context) else {
        return sce_kernel_error(SCE_AJM_ERROR_INVALID_CONTEXT);
    };
    if !context_state.instances.remove(&instance) {
        return sce_kernel_error(SCE_AJM_ERROR_INVALID_INSTANCE);
    }
    eprintln!("[REMu][AJM] destroyed instance={instance:#x} context={context}");
    0
}

fn ajm_finalize(arguments: [u64; 6], state: &mut ProbeDispatchState) -> u64 {
    let context = low_u32(arguments[0]);
    if context == 0 {
        state.ajm_contexts.clear();
    } else if state.ajm_contexts.remove(&context).is_none() {
        return sce_kernel_error(SCE_AJM_ERROR_INVALID_CONTEXT);
    }
    eprintln!("[REMu][AJM] finalized context={context}");
    0
}

fn np_web_api2_initialize(arguments: [u64; 6]) -> u64 {
    const SCE_NP_WEB_API2_ERROR_LIB_CONTEXT_MAX: u32 = 0x8055_3418;
    const MAXIMUM_LIBRARY_CONTEXTS: usize = 0x7fff;

    let context =
        NEXT_NP_WEB_API2_CONTEXT.fetch_update(Ordering::Relaxed, Ordering::Relaxed, |next| {
            (next <= MAXIMUM_LIBRARY_CONTEXTS).then_some(next + 1)
        });
    let Ok(context) = context else {
        return sce_kernel_error(SCE_NP_WEB_API2_ERROR_LIB_CONTEXT_MAX);
    };
    eprintln!(
        "[REMu][NP_WEB_API2] initialized context={context} http_context={} pool_size={:#x}",
        low_i32(arguments[0]),
        arguments[1]
    );
    u64::try_from(context).expect("NP Web API2 context ID fits u64")
}

/// Handles imports whose result depends only on caller-owned memory or thread-local/host state.
///
/// These calls are deliberately kept outside `PROBE_DISPATCH`: Dreaming Sarah invokes the libc
/// memory/string and AGC command-building families from many asset workers, and serializing every
/// one behind the process-wide write lock can stall the render and audio threads even though the
/// calls do not touch `ProbeDispatchState`.
fn dispatch_stateless_import(
    nid: &str,
    arguments: [u64; 6],
    stack_arguments: [u64; 6],
) -> Option<u64> {
    if let Some(result) = agc_command_abi::dispatch_agc_command_abi(nid, arguments, stack_arguments)
    {
        return Some(result);
    }
    Some(match nid {
        "bnZxYgAFeA0" => sanitizer_new_replace_external(),
        "vNe1w4diLCs" => guest_tls_get_addr(arguments[0]),
        "aI+OeCz8xrQ" | "EotR8a3ASf4" => current_guest_thread_handle(),
        "pO96TwzOm5E" => 16 * 1024 * 1024 * 1024,
        "8zTFvBIAIN8" => guest_memset(arguments[0], arguments[1], arguments[2]),
        "Q3VBxCXhUHs" => guest_memcpy(arguments[0], arguments[1], arguments[2]),
        "+P6FRGH4LfA" => guest_memmove(arguments[0], arguments[1], arguments[2]),
        "DfivPArhucg" => guest_memcmp(arguments[0], arguments[1], arguments[2]),
        "8u8lPzUEq+U" => guest_memchr(arguments[0], arguments[1], arguments[2]),
        "fL3O02ypZFE" => guest_wmemcpy(arguments[0], arguments[1], arguments[2]),
        "Noj9PsJrsa8" => guest_wmemmove(arguments[0], arguments[1], arguments[2]),
        "fnUEjBCNRVU" => guest_wmemchr(arguments[0], arguments[1], arguments[2]),
        "QJ5xVfKkni0" => guest_wmemcmp(arguments[0], arguments[1], arguments[2]),
        "2HnmKiLmV6s" => guest_atomic_fetch_4(arguments[0], arguments[1], true),
        "iPBqs+YUUFw" => guest_atomic_fetch_4(arguments[0], arguments[1], false),
        "gQX+4GDQjpM" => guest_malloc(arguments[0]),
        "tIhsqj0qsFE" => guest_free(arguments[0]),
        "2X5agFjKxMc" => guest_calloc(arguments[0], arguments[1]),
        "Y7aJ1uydPMo" => guest_realloc(arguments[0], arguments[1]),
        "Ujf3KzMvRmI" | "2Btkg8k24Zg" => guest_aligned_alloc(arguments[0], arguments[1]),
        "cVSk9y8URbc" => guest_posix_memalign(arguments[0], arguments[1], arguments[2]),
        "j4ViWNHEgww" => guest_strlen(arguments[0]),
        "Ovb2dSJOAuE" => guest_strcmp(arguments[0], arguments[1]),
        "aesyjrHVWy4" => guest_strncmp(arguments[0], arguments[1], arguments[2]),
        "kiZSXIWd9vg" => guest_strcpy(arguments[0], arguments[1]),
        "6sJWiWSRuqk" => guest_strncpy(arguments[0], arguments[1], arguments[2]),
        "Ls4tzzhimqQ" => guest_strcat(arguments[0], arguments[1]),
        "AV6ipCNa4Rw" => guest_strcasecmp(arguments[0], arguments[1]),
        "viiwFMaNamA" => guest_strstr(arguments[0], arguments[1]),
        "rQFVBXp-Cxg" => guest_fseek(arguments[0], arguments[1], arguments[2]),
        "Qazy8LmXTvw" => guest_ftell(arguments[0]),
        "uodLYyUip20" => guest_fclose(arguments[0]),
        "lbB+UlZqVG0" => guest_fread(arguments[0], arguments[1], arguments[2], arguments[3]),
        "kALvdgEv5ME" | "9nf8joUTSaQ" | "Q1BL70XVV0o" => 0,
        "hqi8yMOCmG0" | "p6LrHjIQMdk" | "QW2jL1J5rwY" | "P8F2oavZXtY" => arguments[0],
        "wLlFkwG9UcQ" => guest_time(arguments[0]),
        "tcVi5SivF7Q" => guest_sprintf(arguments),
        "QrZZdJ8XsX0" => guest_fputs(arguments[0]),
        "3GPpjQdAMTw" => guest_guard_acquire(arguments[0]),
        "9rAeANT2tyE" => guest_guard_release(arguments[0]),
        "2emaaluWzUw" => guest_guard_abort(arguments[0]),
        "cfAXurvfl5o" => guest_cxa_allocate_exception(arguments[0]),
        "nOIEswYD4Ig" => guest_cxa_free_exception(arguments[0]),
        "vkuuLfhnSZI" => guest_cxa_throw(arguments),
        "9BcDykPmo1I" => guest_errno_address(),
        "HQOwnfMGipQ" => guest_net_errno_address(),
        "Q4qBuN-c0ZM" => guest_net_unavailable(45),
        "dgJBaeJnGpo" => NEXT_NET_POOL_HANDLE.fetch_add(1, Ordering::Relaxed) as u64,
        "hdpVEUDFW3s" => NEXT_SSL_CONTEXT.fetch_add(1, Ordering::Relaxed) as u64,
        "3JCe3lCbQ8A" | "A9cVMUtEp4Y" | "0gYjPTR-6cY" => {
            NEXT_HTTP_CONTEXT.fetch_add(1, Ordering::Relaxed) as u64
        }
        "+o9816YQhqQ" => np_web_api2_initialize(arguments),
        "MmpF1XsQiHw" => ajm_batch_initialize(arguments),
        "cK6bYHf-Q5E" | "-hJRce8wn1U" => guest_zeroed_constructor(arguments[0], 8),
        "WSOuge5IsCg" => guest_zeroed_constructor(arguments[0], 0x28),
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
        "GIIW2J37e70" => agc_dcb_set_index_size(arguments),
        "B+aG9DUnTKA" => agc_dcb_draw_index_offset(arguments),
        "tSBxhAPyytQ" => agc_dcb_set_num_instances(arguments),
        "Yw0jKSqop+E" => agc_dcb_draw_index_auto(arguments),
        "YUeqkyT7mEQ" => agc_dcb_set_flip(arguments),
        "rPo6tV8D9bM" => system_service_get_status(arguments),
        "fZo48un7LK4" => system_service_param_get_int(arguments),
        "Xjoosiw+XPI" => guest_uuid_create(arguments[0]),
        "yDBwVAolDgg" => guest_kernel_is_stack(arguments),
        "Qs0wWulgl7U" => 0,
        "RaqxZIf6DvE" => 0,
        "rbknaUjpqWo" => write_output_u64(arguments[1], 0),
        "JT+t00a3TxA" => np_get_account_country(arguments[1]),
        "eQH7nWPcAgc" => write_output_i32(arguments[1], 1),
        "e-ZuhGEoeC4" => write_output_i32(arguments[1], 0),
        "1xxcMiGu2fo" => user_service_get_user_name(arguments),
        "woNpu+45RLk" => user_service_get_age_level(arguments),
        "-sD02mFDBh4" => user_service_get_game_presets(arguments),
        "R9lA82OraNs" => app_content_initialize(arguments),
        "99b82IKXpH4" => app_content_app_param_get_int(arguments),
        "eaFXjfJv3xs" | "-4GCfYdNF1s" => 0,
        "18B2NS1y9UU" => rtc_get_current_tick(arguments[0]),
        "ZPD1YOKI+Kw" => rtc_get_current_clock_local_time(arguments[0]),
        "8w-H19ip48I" => rtc_get_tick(arguments),
        "ueega6v3GUw" => rtc_set_tick(arguments),
        "1n37q1Bvc5Y" => system_service_get_display_safe_area_info(arguments[0]),
        "9vA2aW+CHuA" => net_inet_ntop(arguments),
        // The console reports no connection when the title runs offline. Name resolution and
        // network status must fail explicitly so online features disable themselves instead of
        // waiting on a link that never appears.
        "C4UgDHHPvdw" | "obuxdTiwkF8" => guest_net_unavailable(50),
        "nBDD66kiFW8" => 0,
        "zE-wXIZjLoM" => kernel_debug_raise_exception_on_release_mode(arguments),
        "j2AIqSqJP0w" | "2G6i6hMIUUY" => guest_kernel_getdents(arguments, false),
        "taRWhTJFTgE" | "sfKygSjIbI8" => guest_kernel_getdents(arguments, true),
        // Async I/O is initialized with tuning parameters only; REmu services the requests
        // synchronously through the same file paths, so there is no queue to size here.
        "nu4a0-arQis" | "vYU8P9Td2Zo" => 0,
        "EI-5-jlq2dE" => current_guest_thread_handle(),
        "dKadqZFgKKQ" => ime_keyboard_get_resource_ids(arguments),
        "VkqLPArfFdc" => ime_keyboard_get_info(arguments[1]),
        "x8qnXqh-tiM" => 0,
        "rnEhHqG-4xo" | "ZKJtxdgvzwg" | "-3Y5GO+-i78" | "hD-H81EN9Vg" | "O6IW1-Dwm-w" => {
            user_service_get_setting(arguments, 0)
        }
        "qWYHOFwqCxY" => user_service_get_setting(arguments, 1),
        "CdWp0oHWGr0" => write_output_i32(arguments[0], 0x1000_0000),
        "fPhymKNvK-A" => user_service_get_login_user_id_list(arguments),
        "h9z6+0hEydk" => {
            std::thread::yield_now();
            0
        }
        "1jfXLRVzisc" => guest_usleep(arguments[0]),
        "lLMT9vJAck0" | "QBi7HCK03hw" => guest_clock_gettime(arguments[0], arguments[1]),
        "ejekcaNQNq0" => guest_kernel_gettimeofday(arguments[0]),
        "0NTHN1NKONI" => guest_kernel_convert_localtime_to_utc(arguments),
        "-o5uEDpN+oY" => guest_kernel_convert_utc_to_localtime(arguments),
        "0-KXaS70xy4" | "eoht7mQOCmo" => pthread_get_specific(low_u32(arguments[0])),
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
    if let Some(result) = dispatch_stateless_import(nid, arguments, stack_arguments) {
        return result;
    }
    if let Some(result) = dispatch_kernel_file_import(nid, arguments, state) {
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
        "dl+4eHSzUu4" => ajm_initialize(arguments, state),
        "Q3dyFuwGn64" => ajm_module_register(arguments, state),
        "Wi7DtlLV+KI" => ajm_module_unregister(arguments, state),
        "AxoDrINp4J8" => ajm_instance_create(arguments, state),
        "RbLbuKv8zho" => ajm_instance_destroy(arguments, state),
        "MHur6qCsUus" => ajm_finalize(arguments, state),
        // Bundled PRXs are mapped, relocated, and initialized before the main executable. Runtime
        // module loads must still resolve those objects to their stable process-local handles;
        // Unity treats a negative return here as a plug-in load failure and skips assembly setup.
        "LwG8g3niqwA" => kernel_dlsym(arguments, state),
        "WkwEd3N7w0Y" => kernel_install_exception_handler(arguments, state),
        "Qhv5ARAoOEc" => kernel_remove_exception_handler(arguments[0], state),
        "il03nluKfMk" => kernel_raise_exception(arguments, state),
        "6xVpy0Fdq+I" => 0,
        "bnZxYgAFeA0" => sanitizer_new_replace_external(),
        "vNe1w4diLCs" => guest_tls_get_addr(arguments[0]),
        "NWtTN10cJzE" => libc_heap_get_trace_info(arguments[0], state),
        "6c3rCVE-fTU" | "wuCroIGjt2g" => {
            guest_posix_file_result(guest_kernel_open(arguments, state))
        }
        "DRuBt2pvICk" | "AqBioC2vF3I" => guest_posix_file_result(guest_kernel_read(arguments)),
        "NNtFaKJbPt0" | "bY-PO6JhzhQ" => guest_posix_file_result(guest_kernel_close(arguments)),
        "FxVZqBAA7ks" | "FN4gaPmuFV8" => guest_posix_file_result(guest_kernel_write(arguments)),
        "Oy6IpwgtYOk" => guest_posix_file_result(guest_kernel_lseek(arguments)),
        "E6ao34wPw+U" => guest_posix_file_result(guest_kernel_stat(arguments, state)),
        "mqQMh1zPPT8" => guest_posix_file_result(guest_kernel_fstat(arguments)),
        "ih4CD9-gghM" => guest_posix_file_result(guest_kernel_ftruncate(arguments)),
        "4wSze92BhLI" => guest_kernel_write(arguments),
        "aI+OeCz8xrQ" | "EotR8a3ASf4" => current_guest_thread_handle(),
        "7Xl257M4VNI" => u64::from(arguments[0] == arguments[1]),
        // sceKernelGetDirectMemorySize
        "pO96TwzOm5E" => 16 * 1024 * 1024 * 1024,
        // sceKernelAllocateDirectMemory
        "rTXw65xmLIA" => allocate_direct_memory(arguments, state),
        "B+vc2AO2Zrc" => allocate_main_direct_memory(arguments, state),
        "BHouLQzh0X0" => direct_memory_query(arguments, state),
        // sceKernelMapDirectMemory
        "L-Q3LEjIbgA" => map_direct_memory(arguments, state),
        "NcaWUxfMNIQ" => map_direct_memory(arguments, state),
        "mL8NDH86iQI" => map_named_flexible_memory(arguments, state),
        "-hn1tcVHq5Q" => guest_mspace_create(arguments, state),
        "LYo3GhIlB38" => guest_mspace_calloc(arguments, state),
        "OJjm-QOIHlI" => guest_mspace_malloc(arguments, state),
        "Vla-Z+eXlxo" => guest_mspace_free(arguments, state),
        "W6SiVSiCDtI" => guest_mspace_destroy(arguments, state),
        "fEoW6BJsPt4" => guest_mspace_usable_size(arguments, state),
        "gigoVHZvVPE" => guest_mspace_realloc(arguments, state),
        "iF1iQHzxBJU" | "ljkqMcC4-mk" => guest_mspace_memalign(arguments, state),
        "pzUa7KEoydw" => guest_mspace_is_heap_empty(arguments, state),
        "qWESlyXMI3E" => guest_mspace_posix_memalign(arguments, state),
        "I2QC8PYhJWY" => json_init_parameter_set_allocator(arguments, state),
        "Eu95jmqn5Rw" => json_init_parameter_set_file_buffer_size(arguments, state),
        "IXW-z8pggfg" | "Cxwy7wHq4J0" => json_initializer_initialize(arguments, state),
        "PR5k1penBLM" => json_initializer_terminate(arguments, state),
        "eBTreZ84JFY" => av_player_set_log_callback(arguments, state),
        "o9eWRkSL+M4" => av_player_init_ex(arguments, state),
        "HD1YKVU26-M" => av_player_post_init(arguments, state),
        "KMcEa+rHsIo" => av_player_add_source(arguments, state),
        "x8uvuFOPZhU" => av_player_add_source_ex(arguments, state),
        "NkJwDzKmIlw" => av_player_close(arguments[0], state),
        "wwM99gjFf1Y" => av_player_current_time(arguments[0], state),
        "UbQoYawOsfY" => av_player_is_active(arguments[0], state),
        "9y5v+fGN4Wk" => av_player_pause(arguments[0], state),
        "XC9wM+xULz8" => av_player_jump_to_time(arguments, state),
        "w5moABNwnRY" | "ET4Gr-Uu07s" => av_player_start(arguments[0], state),
        "ZC17w3vB5Lo" => av_player_stop(arguments[0], state),
        "OVths0xGfho" => av_player_set_looping(arguments, state),
        "Wnp1OVcrZgk" => av_player_get_audio_data(arguments, state),
        "o3+RWnHViSg" => av_player_get_video_data(arguments, false, state),
        "JdksQu8pNdQ" => av_player_get_video_data(arguments, true, state),
        "d8FcbzfAdQw" => av_player_get_stream_info(arguments, state),
        "ctTAcF5DiKQ" => av_player_get_stream_info(arguments, state),
        "ODJK2sn9w4A" | "k-q+xOxdc3E" => av_player_validate_handle(arguments[0], state),
        "hdTyRzCXQeQ" => av_player_stream_count(arguments[0], state),
        "gEpBkcwxUjw" => ampr_resolve_file_paths(arguments, state),
        "ASoW5WE-UPo" => ampr_submit_command_buffer(arguments, state),
        "rqwFKI4PAiM" => ampr_wait_command_buffer(arguments, state),
        // sceKernelMapDirectMemory2 inserts memory type and passes alignment as argument seven.
        "BQQniolj9tQ" => map_direct_memory_2(arguments, stack_arguments[0], state),
        // Kernel VM ownership calls must describe and retire the mappings created above. Returning
        // success without filling VirtualQuery makes libc feed uninitialized lengths into munmap.
        "rVjRvHJ0X6c" => kernel_virtual_query(arguments, state),
        "cQke9UuBQOk" => kernel_munmap(arguments, state),
        "8zLSfEfW5AU" => coredump_register_handler(arguments, state),
        "2SKEx6bSq-4" => kernel_batch_map(arguments, state),
        "kBJzF8x4SyE" => kernel_batch_map_2(arguments, state),
        "9bfdLIyuwCY" => kernel_mtypeprotect(arguments, state),
        "hwVSPCmp5tM" => kernel_checked_release_direct_memory(arguments, state),
        "MBuItvba6z8" => kernel_release_direct_memory(arguments, state),
        "7oxv3PPCumo" => kernel_reserve_virtual_range(arguments, state),
        // The label is diagnostic metadata only. Validate the guest string so a bad caller still
        // gets an ABI error, but no host VM operation is required.
        "DGMG3JshrZU" => kernel_set_virtual_range_name(arguments),
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
        "1FGvU0i9saQ" => pthread_mutex_attr_set_protocol(arguments, state),
        // The POSIX forms return errno values instead of the signed SCE kernel namespace.
        "dQHWEsJtoE4" => posix_pthread_result(create_pthread_mutex_attr(arguments[0], state)),
        "HF7lK46xzjY" => posix_pthread_result(destroy_pthread_mutex_attr(arguments[0], state)),
        "mDmgMOGVUqg" => posix_pthread_result(pthread_mutex_attr_set_type(arguments, state)),
        "mqULNdimTn0" | "geDaqgH9lTg" => pthread_key_create(arguments, state),
        "WrOLvHU0yQM" | "+BzXYkqYeLE" => pthread_set_specific(arguments, state),
        "PrdHuuDekhY" => pthread_key_delete(low_u32(arguments[0]), state),
        "6BpEZuDT7YI" => pthread_key_delete(low_u32(arguments[0]), state),
        "nsYoNRywwNg" | "wtkt-teR1so" => {
            create_pthread_attr(arguments[0], PthreadAttrState::default(), state)
        }
        "62KCwEMmzcM" | "zHchY8ft5pk" => destroy_pthread_attr(arguments[0], state),
        "-Wreprtu0Qs" => pthread_attr_set_detach_state(arguments, state),
        "UTXzJbWhhTE" | "2Q0z6rnBrTE" => pthread_attr_set_stack_size(arguments, state),
        "DzES9hQF4f4" => pthread_attr_set_schedule_param(arguments, state),
        "FXPWHNk8Of0" => pthread_attr_get_schedule_param(arguments, state),
        "eXbUSpEaTsA" => pthread_attr_set_inherit_schedule(arguments, state),
        "4+h9EzwKF4I" => pthread_attr_set_schedule_policy(arguments, state),
        "P41kTWUS3EI" => pthread_get_schedule_param(arguments, state),
        "oIRFTjoILbg" => pthread_set_schedule_param(arguments, state),
        "bt3CTBKmGyI" => pthread_set_affinity(arguments, state),
        "3qxgM4ezETA" => pthread_attr_set_affinity(arguments, state),
        "W0Hpm2X0uPE" => pthread_set_priority(arguments, state),
        "GBUY7ywdULE" => pthread_rename(arguments, state),
        "x1X76arYMxU" => pthread_attr_get(arguments, state),
        "-quPa4SEJUw" => pthread_attr_get_stack(arguments, state),
        "Ru36fiTtJzA" => pthread_attr_get_stack_address(arguments, state),
        "-fA+7ZlGDQs" => pthread_attr_get_stack_size(arguments, state),
        "8+s5BzZjxSg" => pthread_attr_get_affinity(arguments, state),
        "5txKfcMUAok" => pthread_mutex_attr_set_protocol(arguments, state),
        "m5-2bsNfv7s" => pthread_condition_attr_init(arguments[0], state),
        "waPcxYiR3WA" => pthread_condition_attr_destroy(arguments[0], state),
        // Initial single-device handles for input, video, and audio.
        "xk0AcarP3V4" => pad_open(arguments, state),
        "Up36PTk687E" => video_out_open(arguments, state),
        "utPrVdxio-8" => video_out_get_output_status(arguments, state),
        "CBiu4mCE1DA" => video_out_set_flip_rate(arguments, state),
        "U46NwOiJpys" => video_out_submit_flip(arguments, state),
        "zgXifHT9ErY" => video_out_is_flip_pending(arguments, state),
        "8aI7R7WaOlc" => ampr_command_buffer_constructor(arguments, state),
        "a8uLzYY--tM" => ampr_apr_command_buffer_constructor(arguments, state),
        "N-FSPA4S3nI" => ampr_command_buffer_set_buffer(arguments, state),
        "baQO9ez2gL4" => ampr_command_buffer_reset(arguments, state),
        "ULvXMDz56po" => ampr_command_buffer_clear_buffer(arguments, state),
        "tZDDEo2tE5k" => ampr_command_buffer_get_size(arguments, state),
        "GnxKOHEawhk" => ampr_command_buffer_get_current_offset(arguments, state),
        "H896Pt-yB4I" => ampr_command_buffer_write_kernel_event_queue(arguments, state),
        "sSAUCCU1dv4" | "vWU-odnS+fU" => ampr_measure_command_size(),
        "GuchCTefuZw" | "Qs1xtplKo0U" => ampr_command_buffer_destructor(arguments, state),
        "mQ16-QdKv7k" => ampr_command_buffer_read_file(arguments, stack_arguments, state),
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
        "w2rJhmD+dsE" => agc_driver_add_eq_event(arguments, state),
        "XlNp7jzGiPo" | "MM4IZSEYytQ" => 0,
        "UglJIZjGssM" => agc_driver_submit_dcb(arguments, state),
        "6UzEidRZwkg" => agc_driver_submit_multi_dcbs(arguments, state),
        "gSRnr79F8tQ" => agc_driver_submit_acb(arguments, state),
        "HF3YllT3mXU" => agc_driver_submit_multi_acbs(arguments, state),
        "PjS5uASwcV8" => video_out_set_buffer_attribute2(arguments, state),
        "rKBUtgRrtbk" => video_out_register_buffers2(arguments, state),
        "HXzjK9yI30k" => video_out_add_flip_event(arguments, state),
        "w0hLuNarQxY" => (low_i32(arguments[0]) == 1)
            .then_some(0)
            .unwrap_or_else(|| sce_kernel_error(VIDEO_OUT_ERROR_INVALID_HANDLE)),
        // libScePad output structures must always be initialized; successful no-ops would expose
        // stale guest stack contents as controller capabilities and button state.
        "clVvL4ZDntw" => pad_set_motion_sensor_state(arguments, state),
        "gjP9-KQzoUk" => pad_get_controller_information(arguments, state),
        "6ncge5+l5Qs" => pad_close(arguments, state),
        "AcslpN1jHR8" => pad_get_extended_information(arguments, state),
        "YndgXqQVV7c" => pad_read_state(arguments, state),
        "q1cHNfGycLI" => pad_read(arguments, state),
        "W2G-yoyMF5U" => pad_set_vibration_mode(arguments, state),
        "rIZnR6eSpvk" | "DscD1i9HX1w" => pad_handle_operation(arguments, state),
        "r44mAxdSG+U" | "vDLMoJLde8I" => pad_boolean_operation(arguments, state),
        "yFVnOdGxvZY" | "2JgFB2n9oUM" | "RR4novUEENY" => pad_pointer_operation(arguments, state),
        "znaWI0gpuo8" => pad_get_trigger_effect_state(arguments, state),
        "IHPqcbc0zCA" => pad_device_class_parse_data(arguments, state),
        "hv1luiJrqQM" => pad_initialize(state),
        // libSceSystemService status used by the foreground/background state machine.
        "rPo6tV8D9bM" => system_service_get_status(arguments),
        "fZo48un7LK4" => system_service_param_get_int(arguments),
        "RpQJJVKTiFM" | "4fU5yvOkVG4" => kernel_module_info_for_unwind(arguments, state),
        "crb5j7mkk1c" | "OMDRKKAZ8I4" => 0,
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
        "yH17Q6NWtVg" => user_service_get_event(arguments, state),
        // Runtime services retain initialization and loaded-module state instead of accepting
        // unrelated calls through the unknown-import fallback.
        "g8cM39EUZ6o" => sysmodule_load(arguments[0], state),
        "eR2bZFAAU0Q" => sysmodule_unload(arguments[0], state),
        "fMP5NHUOaMk" => sysmodule_is_loaded(arguments[0], state),
        // PlayGo describes an installed package. REmu runs a fully local image, so every chunk is
        // resident; the calls still validate handles and pointers exactly.
        "ts6GlZOKRrE" => play_go_initialize(state),
        "MPe0EeBGM-E" => play_go_terminate(state),
        "M1Gma1ocrGE" => play_go_open(arguments, state),
        "Uco1I0dlDi8" => play_go_close(arguments[0], state),
        "uWIYLFkkwqk" => play_go_get_locus(arguments, state),
        "-RJWNMK3fC8" => play_go_get_progress(arguments, state),
        "v6EZ-YWRdMs" => play_go_get_eta(arguments, state),
        "rvBSfTimejE" => play_go_get_install_speed(arguments, state),
        "73fF1MFU8hA" => play_go_get_chunk_id(arguments, state),
        "3OMbYZBaa50" => play_go_get_language_mask(arguments, state),
        "4AAcTU9R3XM" | "LosLlHOpNqQ" | "gUPGiOQ1tmQ" | "Nn7zKwnA5q0" | "-Q1-u1a7p0g" => {
            play_go_accept_request(arguments[0], state)
        }
        "lDqxaY1UbEo" => message_dialog_initialize(state),
        "ePw-kqZmelo" => message_dialog_terminate(state),
        "b06Hh0DPEaE" => message_dialog_open(arguments[0], state),
        "HTrcDKlFKuM" => message_dialog_close(state),
        "CWVW78Qc3fI" => message_dialog_get_status(state),
        "6fIC3XKt2k0" => message_dialog_update_status(state),
        "Lr8ovHH9l6A" => message_dialog_get_result(arguments[0], state),
        "buYbeLOGWmA" => app_content_temporary_data_mount(arguments, state),
        "YJtKLttI9fM" => game_update_initialize(state),
        "jO8DM8oyego" => np_entitlement_access_initialize(arguments, state),
        "ysmw6J-P8Ak" => np_session_signaling_initialize(arguments, state),
        "sk54bi6FtYM" => np_web_api2_create_user_context(arguments, state),
        "3Zl8BePTh9Y" => np_check_callback(state),
        "hZIg1EWGsHM" => ult_initialize(arguments[0], state),
        "uoUpLGNkygk" => common_dialog_initialize(state),
        "ERKzksauAJA" => state
            .services
            .save_data_dialog_status
            .cast_unsigned()
            .into(),
        "D0OdFMjp46I" => kernel_create_equeue(arguments[0], state),
        "WDszmSbWuDk" => kernel_add_user_event_edge(arguments, state),
        "bBfz7kMF2Ho" => kernel_add_ampr_event(arguments, state),
        "kwGyyjohI50" => kernel_get_event_data(arguments[0]),
        "Xs9hdiD7sAA" => pthread_set_schedule_parameters(arguments),
        // Direct unit-dispatch callers use the nonblocking poll. Live imports are routed through
        // `kernel_wait_equeue_blocking` before acquiring the process-wide dispatch lock.
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
        "s6W4Zl4Slgk" => np_universal_data_create_property_object(arguments[0], state),
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
        "QBi7HCK03hw" => guest_clock_gettime(arguments[0], arguments[1]),
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

fn kernel_create_equeue(output: u64, state: &mut ProbeDispatchState) -> u64 {
    if output == 0 {
        return sce_kernel_error(SCE_KERNEL_ERROR_MEMORY_FAULT);
    }
    let handle = state.next_equeue.max(1);
    state.next_equeue = handle.saturating_add(1);
    state.kernel_equeues.insert(handle);
    write_guest_u64(output, handle);
    0
}

fn kernel_register_event(
    state: &mut ProbeDispatchState,
    equeue: u64,
    ident: u64,
    filter: i16,
    user_data: u64,
) -> u64 {
    if !state.kernel_equeues.contains(&equeue) {
        return sce_kernel_error(SCE_KERNEL_ERROR_NOT_FOUND);
    }
    state
        .kernel_event_registrations
        .insert((equeue, filter, ident), user_data);
    0
}

fn kernel_add_user_event_edge(arguments: [u64; 6], state: &mut ProbeDispatchState) -> u64 {
    kernel_register_event(
        state,
        arguments[0],
        arguments[1],
        KERNEL_EVENT_FILTER_USER,
        0,
    )
}

/// Reads the `data` field of a delivered kernel event.
///
/// The accessor exists because the event layout is opaque to the title; the field sits after the
/// identifier, filter, flags, and filter flags.
fn kernel_get_event_data(event: u64) -> u64 {
    if event == 0 {
        return 0;
    }
    read_guest_u64(event + 0x10)
}

/// Applies a scheduling policy and priority to a running guest thread.
///
/// REmu runs guest threads on host threads whose priorities the host scheduler owns, so the
/// request is validated and accepted without changing host scheduling. Rejecting it would fail
/// engine worker setup for a property the guest cannot observe directly.
fn pthread_set_schedule_parameters(arguments: [u64; 6]) -> u64 {
    let (thread, parameters) = (arguments[0], arguments[2]);
    if thread == 0 || parameters == 0 {
        return u64::from(libc::EINVAL.cast_unsigned());
    }
    0
}

/// Registers interest in APR/AMPR command-buffer completion on an event queue.
///
/// Registration only expresses interest. Completions are delivered when a command buffer that
/// recorded a queue-write command is submitted; announcing one here would wake the guest's
/// listener for a transfer that never ran, and it would then resolve an event context that does
/// not exist yet.
fn kernel_add_ampr_event(arguments: [u64; 6], state: &mut ProbeDispatchState) -> u64 {
    kernel_register_event(
        state,
        arguments[0],
        arguments[1],
        KERNEL_EVENT_FILTER_AMPR,
        arguments[2],
    )
}

fn agc_driver_add_eq_event(arguments: [u64; 6], state: &mut ProbeDispatchState) -> u64 {
    let result = kernel_register_event(
        state,
        arguments[0],
        arguments[1],
        KERNEL_EVENT_FILTER_GRAPHICS,
        arguments[2],
    );
    if result == 0 {
        eprintln!(
            "[REMu][AGC] registered completion event equeue={:#x} ident={:#x} user_data={:#x}",
            arguments[0], arguments[1], arguments[2]
        );
    }
    result
}

fn kernel_equeue_wake() -> &'static (Mutex<u64>, Condvar) {
    KERNEL_EQUEUE_WAKE.get_or_init(|| (Mutex::new(0), Condvar::new()))
}

fn notify_kernel_equeue_waiters() {
    let wake = kernel_equeue_wake();
    let mut generation = wake
        .0
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    *generation = generation.wrapping_add(1);
    wake.1.notify_all();
}

fn queue_kernel_event(state: &mut ProbeDispatchState, mut event: KernelQueuedEvent) {
    if let Some(pending) = state.pending_events.iter_mut().find(|pending| {
        pending.equeue == event.equeue
            && pending.ident == event.ident
            && pending.filter == event.filter
    }) {
        event.fflags = pending.fflags.saturating_add(1).max(event.fflags);
        *pending = event;
    } else {
        state.pending_events.push_back(event);
    }
}

fn refresh_completed_agc_submissions(state: &mut ProbeDispatchState) {
    let completed = remu_gpu::gpu_completed_submission_sequence();
    let mut queued = false;
    while state
        .pending_agc_completions
        .front()
        .is_some_and(|sequence| *sequence <= completed)
    {
        let sequence = state
            .pending_agc_completions
            .pop_front()
            .expect("the completed AGC sequence was just inspected");
        // The AGC driver completion is not a packet-provided hardware event ID. Titles may use
        // an arbitrary ID when registering their completion queue, so announce end-of-pipe to
        // every graphics registration while preserving that ID in both `ident` and `data`.
        let registrations = state
            .kernel_event_registrations
            .iter()
            .filter_map(|(&(equeue, filter, ident), &user_data)| {
                (filter == KERNEL_EVENT_FILTER_GRAPHICS)
                    .then_some((equeue, ident, filter, user_data))
            })
            .collect::<Vec<_>>();
        for (equeue, ident, filter, user_data) in registrations {
            queue_kernel_event(
                state,
                KernelQueuedEvent {
                    equeue,
                    ident,
                    filter,
                    flags: 0,
                    fflags: 1,
                    data: ident,
                    user_data,
                },
            );
            queued = true;
        }
        if trace_logarithmic_sequence(sequence, 8) {
            eprintln!(
                "[REMu][AGC] completed submission={sequence} backend={completed} equeues={}",
                state
                    .kernel_event_registrations
                    .keys()
                    .filter(|(_, filter, _)| *filter == KERNEL_EVENT_FILTER_GRAPHICS)
                    .count()
            );
        }
    }
    if queued {
        notify_kernel_equeue_waiters();
    }
}

fn kernel_poll_equeue(arguments: [u64; 6], state: &mut ProbeDispatchState) -> Result<u32, u64> {
    let handle = arguments[0];
    let events = arguments[1];
    let event_capacity = low_u32(arguments[2]);
    let out_count = arguments[3];
    if !state.kernel_equeues.contains(&handle) {
        return Err(sce_kernel_error(SCE_KERNEL_ERROR_NOT_FOUND));
    }
    if events == 0 || event_capacity == 0 {
        return Err(sce_kernel_error(SCE_KERNEL_ERROR_INVALID_ARGUMENT));
    }
    refresh_completed_agc_submissions(state);
    let mut delivered = 0_u32;
    while delivered < event_capacity {
        let Some(index) = state
            .pending_events
            .iter()
            .position(|event| event.equeue == handle)
        else {
            break;
        };
        let event = state
            .pending_events
            .remove(index)
            .expect("the queued event index was just found");
        let Some(output) = events.checked_add(u64::from(delivered) * 0x20) else {
            return Err(sce_kernel_error(SCE_KERNEL_ERROR_MEMORY_FAULT));
        };
        write_guest_u64(output, event.ident);
        write_guest_u16(output + 0x08, event.filter.cast_unsigned());
        write_guest_u16(output + 0x0a, event.flags);
        write_guest_u32(output + 0x0c, event.fflags);
        write_guest_u64(output + 0x10, event.data);
        write_guest_u64(output + 0x18, event.user_data);
        if event.filter == KERNEL_EVENT_FILTER_GRAPHICS {
            static GRAPHICS_DELIVERY_COUNT: AtomicUsize = AtomicUsize::new(0);
            let sequence = GRAPHICS_DELIVERY_COUNT.fetch_add(1, Ordering::Relaxed) + 1;
            if sequence <= 64 || sequence.is_power_of_two() {
                eprintln!(
                    "[REMu][AGC][EQUEUE] delivered sequence={sequence} equeue={handle:#x} ident={:#x} flags={:#x} fflags={} data={:#x} user_data={:#x}",
                    event.ident, event.flags, event.fflags, event.data, event.user_data
                );
            }
        }
        delivered += 1;
    }
    if out_count != 0 {
        write_guest_u32(out_count, delivered);
    }
    Ok(delivered)
}

fn kernel_wait_equeue(arguments: [u64; 6], state: &mut ProbeDispatchState) -> u64 {
    kernel_poll_equeue(arguments, state).map_or_else(std::convert::identity, |_| 0)
}

fn kernel_wait_equeue_blocking(arguments: [u64; 6]) -> u64 {
    static WAIT_COUNT: AtomicUsize = AtomicUsize::new(0);
    let wait_sequence = WAIT_COUNT.fetch_add(1, Ordering::Relaxed) + 1;
    let timeout_pointer = arguments[4];
    // Orbis defines the timeout as a pointer to a 32-bit microsecond count. Reading or writing a
    // u64 here overwrites the first four bytes of an adjacent SceKernelEvent when callers keep the
    // timeout immediately before their event array (Unity does exactly that).
    let timeout_microseconds =
        (timeout_pointer != 0).then(|| u64::from(read_guest_u32(timeout_pointer)));
    let deadline = timeout_microseconds
        .and_then(|microseconds| Instant::now().checked_add(Duration::from_micros(microseconds)));
    let guest_thread = current_guest_thread_handle();
    if wait_sequence <= 64 || wait_sequence.is_power_of_two() {
        eprintln!(
            "[REMu][KERNEL][EQUEUE] wait sequence={wait_sequence} equeue={:#x} events={:#x} capacity={} out_count={:#x} timeout_ptr={timeout_pointer:#x} timeout_us={timeout_microseconds:?} thread={guest_thread:#x}",
            arguments[0],
            arguments[1],
            low_u32(arguments[2]),
            arguments[3]
        );
    }
    loop {
        // Snapshot the wake generation without nesting the notifier and dispatch locks. Producers
        // hold the dispatch lock while enqueueing, so a nested inverse order could deadlock.
        let generation = *kernel_equeue_wake()
            .0
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let result = {
            let state = PROBE_DISPATCH.get_or_init(|| RwLock::new(ProbeDispatchState::default()));
            let mut state = state
                .write()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            kernel_poll_equeue(arguments, &mut state)
        };
        match result {
            Err(error) => return error,
            Ok(count) if count != 0 => {
                if let (Some(deadline), true) = (deadline, timeout_pointer != 0) {
                    let remaining = deadline
                        .checked_duration_since(Instant::now())
                        .map_or(0, |duration| {
                            u64::try_from(duration.as_micros()).unwrap_or(u64::MAX)
                        });
                    write_guest_u32(
                        timeout_pointer,
                        u32::try_from(remaining).unwrap_or(u32::MAX),
                    );
                }
                if wait_sequence <= 64 || wait_sequence.is_power_of_two() {
                    eprintln!(
                        "[REMu][KERNEL][EQUEUE] woke sequence={wait_sequence} equeue={:#x} delivered={count}",
                        arguments[0]
                    );
                }
                return 0;
            }
            Ok(_) => {}
        }
        if guest_exception_is_pending(guest_thread) {
            return sce_kernel_error(SCE_KERNEL_ERROR_INTERRUPTED);
        }
        let wait_for = if let Some(deadline) = deadline {
            let Some(remaining) = deadline.checked_duration_since(Instant::now()) else {
                write_guest_u32(timeout_pointer, 0);
                return sce_kernel_error(SCE_KERNEL_ERROR_TIMED_OUT);
            };
            remaining.min(Duration::from_millis(1))
        } else {
            // Backend submission completion is an atomic published from the presenter. Polling it
            // at a 1 ms ceiling avoids a cross-crate callback while keeping idle waits parked.
            Duration::from_millis(1)
        };
        let wake = kernel_equeue_wake();
        let wake_state = wake
            .0
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        if *wake_state != generation {
            continue;
        }
        let _ = wake
            .1
            .wait_timeout(wake_state, wait_for)
            .unwrap_or_else(std::sync::PoisonError::into_inner);
    }
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

fn sysmodule_unload(module: u64, state: &mut ProbeDispatchState) -> u64 {
    const SYSMODULE_INVALID_ID: u32 = 0x805a_1000;
    const SYSMODULE_NOT_LOADED: u32 = 0x805a_1001;
    let module = low_i32(module);
    if module <= 0 {
        return sce_kernel_error(SYSMODULE_INVALID_ID);
    }
    if !state.services.loaded_sysmodules.remove(&module) {
        return sce_kernel_error(SYSMODULE_NOT_LOADED);
    }
    0
}

fn sysmodule_is_loaded(module: u64, state: &ProbeDispatchState) -> u64 {
    const SYSMODULE_INVALID_ID: u32 = 0x805a_1000;
    const SYSMODULE_NOT_LOADED: u32 = 0x805a_1001;
    let module = low_i32(module);
    if module <= 0 {
        return sce_kernel_error(SYSMODULE_INVALID_ID);
    }
    if state.services.loaded_sysmodules.contains(&module) {
        0
    } else {
        sce_kernel_error(SYSMODULE_NOT_LOADED)
    }
}

fn game_update_initialize(state: &mut ProbeDispatchState) -> u64 {
    const GAME_UPDATE_SYSMODULE: i32 = 0xcc;
    if !state
        .services
        .loaded_sysmodules
        .contains(&GAME_UPDATE_SYSMODULE)
    {
        return sce_kernel_error(0x805a_1001);
    }
    state.services.game_update = InitializationState::Initialized;
    0
}

fn np_entitlement_access_initialize(arguments: [u64; 6], state: &mut ProbeDispatchState) -> u64 {
    if arguments[0] == 0 || arguments[1] == 0 {
        return sce_kernel_error(SCE_KERNEL_ERROR_INVALID_ARGUMENT);
    }
    state.services.np_entitlement_access = Some((arguments[0], arguments[1]));
    0
}

fn np_session_signaling_initialize(arguments: [u64; 6], state: &mut ProbeDispatchState) -> u64 {
    const PARAMETER_SIZE: usize = 0x28;
    if arguments[0] == 0 {
        return sce_kernel_error(SCE_KERNEL_ERROR_INVALID_ARGUMENT);
    }
    // SAFETY: this ABI takes a fixed-size initialization structure that remains caller-owned.
    let parameter =
        unsafe { std::slice::from_raw_parts(arguments[0] as *const u8, PARAMETER_SIZE).to_vec() };
    state.services.np_session_signaling = Some(parameter);
    0
}

fn np_web_api2_create_user_context(arguments: [u64; 6], state: &mut ProbeDispatchState) -> u64 {
    const INVALID_ARGUMENT: u32 = 0x8055_3402;
    const INVALID_LIBRARY_CONTEXT: u32 = 0x8055_3403;
    const USER_CONTEXT_ALREADY_EXISTS: u32 = 0x8055_340a;
    let (library_context, user_id) = (low_i32(arguments[0]), low_i32(arguments[1]));
    if !(1..0x8000).contains(&library_context) {
        return sce_kernel_error(INVALID_LIBRARY_CONTEXT);
    }
    if user_id == -1 {
        return sce_kernel_error(INVALID_ARGUMENT);
    }
    if state
        .services
        .np_web_api2_user_contexts
        .values()
        .any(|&(library, user)| library == library_context && user == user_id)
    {
        return sce_kernel_error(USER_CONTEXT_ALREADY_EXISTS);
    }
    let context = state.services.next_np_web_api2_user_context.max(1);
    state.services.next_np_web_api2_user_context = context.saturating_add(1);
    state
        .services
        .np_web_api2_user_contexts
        .insert(context, (library_context, user_id));
    signed_i32_return(context)
}

fn np_check_callback(state: &mut ProbeDispatchState) -> u64 {
    state.services.np_callback_polls = state.services.np_callback_polls.saturating_add(1);
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

fn user_service_get_event(arguments: [u64; 6], state: &mut ProbeDispatchState) -> u64 {
    const USER_SERVICE_ERROR_INVALID_ARGUMENT: u32 = 0x8096_0005;
    const USER_SERVICE_ERROR_NO_EVENT: u32 = 0x8096_0007;
    if arguments[0] == 0 {
        return sce_kernel_error(USER_SERVICE_ERROR_INVALID_ARGUMENT);
    }
    if state.services.user_login_event_delivered {
        return sce_kernel_error(USER_SERVICE_ERROR_NO_EVENT);
    }
    write_guest_u32(arguments[0], 0);
    write_guest_u32(arguments[0] + 4, 0x1000_0000);
    state.services.user_login_event_delivered = true;
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

fn np_get_account_country(output: u64) -> u64 {
    if output == 0 {
        return sce_kernel_error(SCE_KERNEL_ERROR_INVALID_ARGUMENT);
    }
    // SceNpCountryCode is two ISO-3166 letters followed by a terminator and padding. Network
    // state is reported separately as signed out, but callers still expect deterministic profile
    // metadata rather than uninitialized stack bytes.
    write_guest_bytes(output, b"US\0\0");
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

fn pad_close(arguments: [u64; 6], state: &ProbeDispatchState) -> u64 {
    validate_pad_handle(arguments[0], state).map_or_else(|error| error, |()| 0)
}

fn pad_get_extended_information(arguments: [u64; 6], state: &ProbeDispatchState) -> u64 {
    if let Err(error) = validate_pad_handle(arguments[0], state) {
        return error;
    }
    if arguments[1] == 0 {
        return sce_kernel_error(SCE_KERNEL_ERROR_INVALID_ARGUMENT);
    }
    // A standard DualSense has no special device class payload. The ABI structure is 20 bytes:
    // class, reserved word, and a 12-byte class-specific union.
    guest_memset(arguments[1], 0, 20);
    0
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

fn pad_set_vibration_mode(arguments: [u64; 6], state: &ProbeDispatchState) -> u64 {
    if let Err(error) = validate_pad_handle(arguments[0], state) {
        return error;
    }
    if low_u32(arguments[1]) > 2 {
        return sce_kernel_error(SCE_KERNEL_ERROR_INVALID_ARGUMENT);
    }
    0
}

fn pad_handle_operation(arguments: [u64; 6], state: &ProbeDispatchState) -> u64 {
    validate_pad_handle(arguments[0], state).map_or_else(|error| error, |()| 0)
}

fn pad_boolean_operation(arguments: [u64; 6], state: &ProbeDispatchState) -> u64 {
    if let Err(error) = validate_pad_handle(arguments[0], state) {
        return error;
    }
    if !matches!(low_u32(arguments[1]), 0 | 1) {
        return sce_kernel_error(SCE_KERNEL_ERROR_INVALID_ARGUMENT);
    }
    0
}

fn pad_pointer_operation(arguments: [u64; 6], state: &ProbeDispatchState) -> u64 {
    if let Err(error) = validate_pad_handle(arguments[0], state) {
        return error;
    }
    if arguments[1] == 0 {
        return sce_kernel_error(SCE_KERNEL_ERROR_INVALID_ARGUMENT);
    }
    0
}

fn pad_get_trigger_effect_state(arguments: [u64; 6], state: &ProbeDispatchState) -> u64 {
    if let Err(error) = validate_pad_handle(arguments[0], state) {
        return error;
    }
    if arguments[1] == 0 {
        return sce_kernel_error(SCE_KERNEL_ERROR_INVALID_ARGUMENT);
    }
    guest_memset(arguments[1], 0, 8);
    0
}

fn pad_device_class_parse_data(arguments: [u64; 6], state: &ProbeDispatchState) -> u64 {
    if let Err(error) = validate_pad_handle(arguments[0], state) {
        return error;
    }
    if arguments[1] == 0 || arguments[2] == 0 {
        return sce_kernel_error(SCE_KERNEL_ERROR_INVALID_ARGUMENT);
    }
    guest_memset(arguments[2], 0, 24);
    write_guest_u8(arguments[2] + 4, read_guest_u8(arguments[1] + 0x4c));
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

fn pad_read(arguments: [u64; 6], state: &ProbeDispatchState) -> u64 {
    if arguments[2] == 0 || arguments[2] > 64 {
        return sce_kernel_error(SCE_KERNEL_ERROR_INVALID_ARGUMENT);
    }
    let result = pad_read_state(arguments, state);
    if result == 0 { 1 } else { result }
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

fn guest_uuid_create(output: u64) -> u64 {
    static NEXT_UUID: AtomicU32 = AtomicU32::new(1);
    if output == 0 {
        return sce_kernel_error(SCE_KERNEL_ERROR_INVALID_ARGUMENT);
    }
    let sequence = u64::from(NEXT_UUID.fetch_add(1, Ordering::Relaxed));
    let seconds = current_unix_seconds().cast_unsigned();
    let mut bytes = [0_u8; 16];
    bytes[..8].copy_from_slice(&(seconds.rotate_left(17) ^ sequence).to_le_bytes());
    bytes[8..]
        .copy_from_slice(&(sequence.wrapping_mul(0x9e37_79b9_7f4a_7c15) ^ output).to_le_bytes());
    bytes[6] = (bytes[6] & 0x0f) | 0x40;
    bytes[8] = (bytes[8] & 0x3f) | 0x80;
    write_guest_bytes(output, &bytes);
    0
}

fn guest_kernel_is_stack(arguments: [u64; 6]) -> u64 {
    if arguments[1] != 0 {
        write_guest_u64(arguments[1], 0);
    }
    if arguments[2] != 0 {
        write_guest_u64(arguments[2], 0);
    }
    0
}

fn user_service_get_user_name(arguments: [u64; 6]) -> u64 {
    const USER_SERVICE_ERROR_INVALID_ARGUMENT: u32 = 0x8096_0005;
    const USER_SERVICE_ERROR_INVALID_PARAMETER: u32 = 0x8096_0009;
    const USER_SERVICE_ERROR_BUFFER_TOO_SHORT: u32 = 0x8096_000a;
    const USER_NAME: &[u8] = b"REmu\0";

    if low_i32(arguments[0]) != 0x1000_0000 {
        return sce_kernel_error(USER_SERVICE_ERROR_INVALID_PARAMETER);
    }
    if arguments[1] == 0 {
        return sce_kernel_error(USER_SERVICE_ERROR_INVALID_ARGUMENT);
    }
    if arguments[2] < USER_NAME.len() as u64 {
        return sce_kernel_error(USER_SERVICE_ERROR_BUFFER_TOO_SHORT);
    }
    write_guest_bytes(arguments[1], USER_NAME);
    0
}

fn user_service_get_age_level(arguments: [u64; 6]) -> u64 {
    const USER_SERVICE_ERROR_INVALID_ARGUMENT: u32 = 0x8096_0005;
    const USER_SERVICE_ERROR_INVALID_PARAMETER: u32 = 0x8096_0009;
    if low_i32(arguments[0]) != 0x1000_0000 {
        return sce_kernel_error(USER_SERVICE_ERROR_INVALID_PARAMETER);
    }
    if arguments[1] == 0 {
        return sce_kernel_error(USER_SERVICE_ERROR_INVALID_ARGUMENT);
    }
    write_guest_u32(arguments[1], 18);
    0
}

fn user_service_get_game_presets(arguments: [u64; 6]) -> u64 {
    const USER_SERVICE_ERROR_INVALID_ARGUMENT: u32 = 0x8096_0005;
    const USER_SERVICE_ERROR_INVALID_PARAMETER: u32 = 0x8096_0009;
    if low_i32(arguments[0]) != 0x1000_0000 {
        return sce_kernel_error(USER_SERVICE_ERROR_INVALID_PARAMETER);
    }
    if arguments[1] == 0 {
        return sce_kernel_error(USER_SERVICE_ERROR_INVALID_ARGUMENT);
    }
    guest_memset(arguments[1], 0, 0x28);
    write_guest_u64(arguments[1], 0x28);
    0
}

fn user_service_get_setting(arguments: [u64; 6], value: u32) -> u64 {
    const USER_SERVICE_ERROR_INVALID_ARGUMENT: u32 = 0x8096_0005;
    const USER_SERVICE_ERROR_INVALID_PARAMETER: u32 = 0x8096_0009;
    if low_i32(arguments[0]) != 0x1000_0000 {
        return sce_kernel_error(USER_SERVICE_ERROR_INVALID_PARAMETER);
    }
    if arguments[1] == 0 {
        return sce_kernel_error(USER_SERVICE_ERROR_INVALID_ARGUMENT);
    }
    write_guest_u32(arguments[1], value);
    0
}

fn app_content_initialize(arguments: [u64; 6]) -> u64 {
    if arguments[0] == 0 || arguments[1] == 0 {
        return sce_kernel_error(SCE_KERNEL_ERROR_INVALID_ARGUMENT);
    }
    write_guest_u32(arguments[1] + 4, 0);
    0
}

fn app_content_app_param_get_int(arguments: [u64; 6]) -> u64 {
    if arguments[1] == 0 || low_u32(arguments[0]) > 4 {
        return sce_kernel_error(SCE_KERNEL_ERROR_INVALID_ARGUMENT);
    }
    let value = if low_u32(arguments[0]) == 0 { 3 } else { 0 };
    write_guest_u32(arguments[1], value);
    0
}

/// Decomposes a tick count into an `SceRtcDateTime`.
///
/// This is the inverse of [`rtc_get_tick`] and the accessor engines use to turn a stored
/// timestamp back into a calendar date. Leaving the structure untouched leaves the caller with
/// whatever the stack held, which fails date validation with implausible field values.
fn rtc_set_tick(arguments: [u64; 6]) -> u64 {
    let (output, input) = (arguments[0], arguments[1]);
    if output == 0 || input == 0 {
        return sce_kernel_error(RTC_ERROR_INVALID_POINTER);
    }
    let ticks = read_guest_u64(input);
    let epoch_ticks = RTC_UNIX_EPOCH_TICK_MICROSECONDS;
    // Ticks count from year one; convert to the Unix origin the civil decomposition uses.
    let unix_microseconds = i128::from(ticks) - i128::from(epoch_ticks);
    let seconds = i64::try_from(unix_microseconds.div_euclid(1_000_000)).unwrap_or(0);
    let microsecond = u32::try_from(unix_microseconds.rem_euclid(1_000_000)).unwrap_or(0);
    let (year, month, day, hour, minute, second) = civil_from_unix_seconds(seconds);
    if year == 0 {
        return sce_kernel_error(RTC_ERROR_INVALID_ARGUMENT);
    }
    write_guest_u16(output, year);
    write_guest_u16(output + 2, month);
    write_guest_u16(output + 4, day);
    write_guest_u16(output + 6, hour);
    write_guest_u16(output + 8, minute);
    write_guest_u16(output + 10, second);
    write_guest_u32(output + 12, microsecond);
    0
}

/// Reports the display's safe area as the full display.
///
/// The console lets users shrink the safe area; REmu presents the guest's full requested surface,
/// so the ratio is one and the insets are zero.
fn system_service_get_display_safe_area_info(output: u64) -> u64 {
    if output == 0 {
        return sce_kernel_error(SCE_KERNEL_ERROR_INVALID_ARGUMENT);
    }
    guest_memset(output, 0, 32);
    write_guest_u32(output, 1.0_f32.to_bits());
    0
}

/// Converts a binary network address into its textual form.
fn net_inet_ntop(arguments: [u64; 6]) -> u64 {
    const AF_INET: i32 = 2;
    const AF_INET6: i32 = 28;
    let (family, source, destination, capacity) = (
        low_i32(arguments[0]),
        arguments[1],
        arguments[2],
        arguments[3],
    );
    if source == 0 || destination == 0 {
        return 0;
    }
    let text = match family {
        AF_INET => {
            let octets = [
                read_guest_u8(source),
                read_guest_u8(source + 1),
                read_guest_u8(source + 2),
                read_guest_u8(source + 3),
            ];
            format!("{}.{}.{}.{}", octets[0], octets[1], octets[2], octets[3])
        }
        AF_INET6 => {
            let groups: Vec<String> = (0..8)
                .map(|group| format!("{:x}", read_guest_u16(source + group * 2).to_be()))
                .collect();
            groups.join(":")
        }
        _ => return 0,
    };
    if capacity <= text.len() as u64 {
        return 0;
    }
    write_guest_bytes(destination, text.as_bytes());
    write_guest_bytes(destination + text.len() as u64, b"\0");
    destination
}

/// Converts an `SceRtcDateTime` into a tick count.
///
/// Ticks are microseconds since the start of year one, which is the same origin
/// `sceRtcGetCurrentTick` reports, so values from both calls are directly comparable.
fn rtc_get_tick(arguments: [u64; 6]) -> u64 {
    let (input, output) = (arguments[0], arguments[1]);
    if input == 0 || output == 0 {
        return sce_kernel_error(RTC_ERROR_INVALID_POINTER);
    }
    let year = i64::from(read_guest_u16(input));
    let month = i64::from(read_guest_u16(input + 2));
    let day = i64::from(read_guest_u16(input + 4));
    let hour = i64::from(read_guest_u16(input + 6));
    let minute = i64::from(read_guest_u16(input + 8));
    let second = i64::from(read_guest_u16(input + 10));
    let microsecond = i64::from(read_guest_u32(input + 12));
    if !(1..=9999).contains(&year)
        || !(1..=12).contains(&month)
        || !(1..=31).contains(&day)
        || hour > 23
        || minute > 59
        || second > 59
    {
        return sce_kernel_error(RTC_ERROR_INVALID_ARGUMENT);
    }
    let days = unix_days_from_civil(year, month, day);
    let seconds = days * 86_400 + hour * 3_600 + minute * 60 + second;
    let ticks = i64::try_from(RTC_UNIX_EPOCH_TICK_MICROSECONDS).unwrap_or(i64::MAX)
        + seconds.saturating_mul(1_000_000)
        + microsecond;
    write_guest_u64(output, u64::from_ne_bytes(ticks.to_ne_bytes()));
    0
}

/// Days from the Unix epoch to a proleptic Gregorian date; the inverse of [`civil_from_unix_seconds`].
fn unix_days_from_civil(year: i64, month: i64, day: i64) -> i64 {
    let year = year - i64::from(month <= 2);
    let era = year.div_euclid(400);
    let year_of_era = year - era * 400;
    let shifted_month = if month > 2 { month - 3 } else { month + 9 };
    let day_of_year = (153 * shifted_month + 2) / 5 + day - 1;
    let day_of_era = year_of_era * 365 + year_of_era / 4 - year_of_era / 100 + day_of_year;
    era * 146_097 + day_of_era - 719_468
}

/// Registers the writable per-title temporary data directory.
fn app_content_temporary_data_mount(arguments: [u64; 6], state: &ProbeDispatchState) -> u64 {
    let output = arguments[1];
    if output == 0 {
        return sce_kernel_error(SCE_KERNEL_ERROR_INVALID_ARGUMENT);
    }
    let mount = state.save_root.join("temp0");
    if let Err(error) = std::fs::create_dir_all(&mount) {
        eprintln!(
            "[REMu][VFS] temporary data mount failed {:?}: {error}",
            mount.display()
        );
        return sce_kernel_io_error(&error);
    }
    // The console reports the guest-visible mount point, not a host path.
    write_guest_bytes(output, b"/temp0\0");
    0
}

/// Terminates the process for a guest-raised release-mode exception.
///
/// The title calls this after its own fatal-error handling has already run; continuing would let
/// the guest execute past an error it declared unrecoverable.
fn kernel_debug_raise_exception_on_release_mode(arguments: [u64; 6]) -> u64 {
    eprintln!(
        "[REMu][GUEST] fatal guest exception error={:#x} argument={:#x}",
        low_u32(arguments[0]),
        arguments[1]
    );
    terminate_probe(70)
}

fn rtc_get_current_tick(output: u64) -> u64 {
    if output == 0 {
        return sce_kernel_error(RTC_ERROR_INVALID_POINTER);
    }
    let unix_microseconds = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |duration| {
            duration
                .as_secs()
                .saturating_mul(1_000_000)
                .saturating_add(u64::from(duration.subsec_micros()))
        });
    write_guest_u64(
        output,
        RTC_UNIX_EPOCH_TICK_MICROSECONDS.saturating_add(unix_microseconds),
    );
    0
}

const PLAYGO_ERROR_NOT_INITIALIZED: u32 = 0x80b2_0002;
const PLAYGO_ERROR_ALREADY_INITIALIZED: u32 = 0x80b2_0004;
const PLAYGO_ERROR_BAD_HANDLE: u32 = 0x80b2_0006;
const PLAYGO_ERROR_BAD_POINTER: u32 = 0x80b2_0008;
const PLAYGO_ERROR_BAD_SIZE: u32 = 0x80b2_0009;
/// The only handle the console hands out for the running application's PlayGo package.
const PLAYGO_HANDLE: u32 = 1;
/// Chunk is resident on local storage with fast access. REmu runs from a fully installed local
/// image, so every chunk the title asks about is already present.
const PLAYGO_LOCUS_LOCAL_FAST: u32 = 3;
const PLAYGO_INSTALL_SPEED_FULL: u32 = 1;

/// Reports whether PlayGo is initialized, rejecting the double-initialize the ABI forbids.
fn play_go_initialize(state: &mut ProbeDispatchState) -> u64 {
    if state.services.play_go_initialized {
        return sce_kernel_error(PLAYGO_ERROR_ALREADY_INITIALIZED);
    }
    state.services.play_go_initialized = true;
    0
}

fn play_go_terminate(state: &mut ProbeDispatchState) -> u64 {
    if !state.services.play_go_initialized {
        return sce_kernel_error(PLAYGO_ERROR_NOT_INITIALIZED);
    }
    state.services.play_go_initialized = false;
    state.services.play_go_open = false;
    0
}

fn play_go_open(arguments: [u64; 6], state: &mut ProbeDispatchState) -> u64 {
    let output = arguments[0];
    if !state.services.play_go_initialized {
        return sce_kernel_error(PLAYGO_ERROR_NOT_INITIALIZED);
    }
    if output == 0 {
        return sce_kernel_error(PLAYGO_ERROR_BAD_POINTER);
    }
    state.services.play_go_open = true;
    write_guest_u32(output, PLAYGO_HANDLE);
    0
}

fn play_go_close(handle: u64, state: &mut ProbeDispatchState) -> u64 {
    if low_u32(handle) != PLAYGO_HANDLE || !state.services.play_go_open {
        return sce_kernel_error(PLAYGO_ERROR_BAD_HANDLE);
    }
    state.services.play_go_open = false;
    0
}

/// Validates the handle and entry count shared by every per-chunk PlayGo query.
fn play_go_entry_query(
    handle: u64,
    chunk_ids: u64,
    entries: u64,
    output: u64,
    state: &ProbeDispatchState,
) -> Result<u32, u64> {
    if !state.services.play_go_initialized {
        return Err(sce_kernel_error(PLAYGO_ERROR_NOT_INITIALIZED));
    }
    if low_u32(handle) != PLAYGO_HANDLE {
        return Err(sce_kernel_error(PLAYGO_ERROR_BAD_HANDLE));
    }
    if chunk_ids == 0 || output == 0 {
        return Err(sce_kernel_error(PLAYGO_ERROR_BAD_POINTER));
    }
    let entries = low_u32(entries);
    if entries == 0 {
        return Err(sce_kernel_error(PLAYGO_ERROR_BAD_SIZE));
    }
    Ok(entries)
}

fn play_go_get_locus(arguments: [u64; 6], state: &ProbeDispatchState) -> u64 {
    let entries = match play_go_entry_query(
        arguments[0],
        arguments[1],
        arguments[2],
        arguments[3],
        state,
    ) {
        Ok(entries) => entries,
        Err(error) => return error,
    };
    for entry in 0..u64::from(entries) {
        write_guest_u32(arguments[3] + entry * 4, PLAYGO_LOCUS_LOCAL_FAST);
    }
    0
}

/// Reports installation progress. A locally installed image is always fully downloaded, so the
/// progressed and total sizes are equal and non-zero; a zero total reads as "nothing installed".
fn play_go_get_progress(arguments: [u64; 6], state: &ProbeDispatchState) -> u64 {
    let entries = match play_go_entry_query(
        arguments[0],
        arguments[1],
        arguments[2],
        arguments[3],
        state,
    ) {
        Ok(entries) => entries,
        Err(error) => return error,
    };
    // The console reports byte counts per requested chunk; the exact size is unobservable to the
    // title beyond progressed == total, which is what "installed" means.
    let total = u64::from(entries) * GUEST_PAGE_SIZE;
    write_guest_u64(arguments[3], total);
    write_guest_u64(arguments[3] + 8, total);
    0
}

fn play_go_get_eta(arguments: [u64; 6], state: &ProbeDispatchState) -> u64 {
    let _ = match play_go_entry_query(
        arguments[0],
        arguments[1],
        arguments[2],
        arguments[3],
        state,
    ) {
        Ok(entries) => entries,
        Err(error) => return error,
    };
    // Nothing remains to be downloaded.
    write_guest_u64(arguments[3], 0);
    0
}

fn play_go_get_install_speed(arguments: [u64; 6], state: &ProbeDispatchState) -> u64 {
    if !state.services.play_go_initialized {
        return sce_kernel_error(PLAYGO_ERROR_NOT_INITIALIZED);
    }
    if low_u32(arguments[0]) != PLAYGO_HANDLE {
        return sce_kernel_error(PLAYGO_ERROR_BAD_HANDLE);
    }
    if arguments[1] == 0 {
        return sce_kernel_error(PLAYGO_ERROR_BAD_POINTER);
    }
    write_guest_u32(arguments[1], PLAYGO_INSTALL_SPEED_FULL);
    0
}

/// Reports the chunk identifiers belonging to the installed image.
///
/// A locally installed title is delivered as one chunk, so the list contains only chunk zero.
fn play_go_get_chunk_id(arguments: [u64; 6], state: &ProbeDispatchState) -> u64 {
    let (output, entries, count_output) = (arguments[1], low_u32(arguments[2]), arguments[3]);
    if !state.services.play_go_initialized {
        return sce_kernel_error(PLAYGO_ERROR_NOT_INITIALIZED);
    }
    if low_u32(arguments[0]) != PLAYGO_HANDLE {
        return sce_kernel_error(PLAYGO_ERROR_BAD_HANDLE);
    }
    if output == 0 || count_output == 0 {
        return sce_kernel_error(PLAYGO_ERROR_BAD_POINTER);
    }
    if entries == 0 {
        return sce_kernel_error(PLAYGO_ERROR_BAD_SIZE);
    }
    write_guest_u32(output, 0);
    write_guest_u32(count_output, 1);
    0
}

/// Reports the language mask selected for the package. Zero means no language filtering.
fn play_go_get_language_mask(arguments: [u64; 6], state: &ProbeDispatchState) -> u64 {
    if !state.services.play_go_initialized {
        return sce_kernel_error(PLAYGO_ERROR_NOT_INITIALIZED);
    }
    if low_u32(arguments[0]) != PLAYGO_HANDLE {
        return sce_kernel_error(PLAYGO_ERROR_BAD_HANDLE);
    }
    if arguments[1] == 0 {
        return sce_kernel_error(PLAYGO_ERROR_BAD_POINTER);
    }
    write_guest_u64(arguments[1], 0);
    0
}

/// Accepts a download-ordering or speed request against an already complete installation.
fn play_go_accept_request(handle: u64, state: &ProbeDispatchState) -> u64 {
    if !state.services.play_go_initialized {
        return sce_kernel_error(PLAYGO_ERROR_NOT_INITIALIZED);
    }
    if low_u32(handle) != PLAYGO_HANDLE {
        return sce_kernel_error(PLAYGO_ERROR_BAD_HANDLE);
    }
    0
}

const COMMON_DIALOG_STATUS_NONE: u32 = 0;
const COMMON_DIALOG_STATUS_INITIALIZED: u32 = 1;
const COMMON_DIALOG_STATUS_RUNNING: u32 = 2;
const COMMON_DIALOG_STATUS_FINISHED: u32 = 3;
const COMMON_DIALOG_ERROR_NOT_INITIALIZED: u32 = 0x8002_0001;
const COMMON_DIALOG_ERROR_ALREADY_INITIALIZED: u32 = 0x8002_0011;
const COMMON_DIALOG_ERROR_ARG_NULL: u32 = 0x8002_000e;
/// The dialog closed without the user choosing a button.
const MESSAGE_DIALOG_BUTTON_INVALID: u32 = 0;
const MESSAGE_DIALOG_RESULT_USER_CANCELED: u32 = 1;

fn message_dialog_initialize(state: &mut ProbeDispatchState) -> u64 {
    if state.services.message_dialog_status != COMMON_DIALOG_STATUS_NONE {
        return sce_kernel_error(COMMON_DIALOG_ERROR_ALREADY_INITIALIZED);
    }
    state.services.message_dialog_status = COMMON_DIALOG_STATUS_INITIALIZED;
    0
}

fn message_dialog_terminate(state: &mut ProbeDispatchState) -> u64 {
    if state.services.message_dialog_status == COMMON_DIALOG_STATUS_NONE {
        return sce_kernel_error(COMMON_DIALOG_ERROR_NOT_INITIALIZED);
    }
    state.services.message_dialog_status = COMMON_DIALOG_STATUS_NONE;
    0
}

/// Opens a system message dialog.
///
/// REmu presents no system dialog surface, so the dialog is immediately dismissed rather than
/// left running: a title that polls `sceMsgDialogUpdateStatus` until the dialog finishes would
/// otherwise never make progress. The recorded result reports that the user canceled without
/// choosing a button, which is the outcome a dismissed dialog produces on the console.
fn message_dialog_open(parameters: u64, state: &mut ProbeDispatchState) -> u64 {
    if state.services.message_dialog_status == COMMON_DIALOG_STATUS_NONE {
        return sce_kernel_error(COMMON_DIALOG_ERROR_NOT_INITIALIZED);
    }
    if parameters == 0 {
        return sce_kernel_error(COMMON_DIALOG_ERROR_ARG_NULL);
    }
    state.services.message_dialog_status = COMMON_DIALOG_STATUS_RUNNING;
    0
}

fn message_dialog_update_status(state: &mut ProbeDispatchState) -> u64 {
    if state.services.message_dialog_status == COMMON_DIALOG_STATUS_RUNNING {
        state.services.message_dialog_status = COMMON_DIALOG_STATUS_FINISHED;
    }
    u64::from(state.services.message_dialog_status)
}

fn message_dialog_get_status(state: &ProbeDispatchState) -> u64 {
    u64::from(state.services.message_dialog_status)
}

fn message_dialog_close(state: &mut ProbeDispatchState) -> u64 {
    if state.services.message_dialog_status == COMMON_DIALOG_STATUS_NONE {
        return sce_kernel_error(COMMON_DIALOG_ERROR_NOT_INITIALIZED);
    }
    state.services.message_dialog_status = COMMON_DIALOG_STATUS_FINISHED;
    0
}

/// Fills `SceMsgDialogResult`: mode, result code, selected button, then reserved bytes.
fn message_dialog_get_result(output: u64, state: &ProbeDispatchState) -> u64 {
    if state.services.message_dialog_status == COMMON_DIALOG_STATUS_NONE {
        return sce_kernel_error(COMMON_DIALOG_ERROR_NOT_INITIALIZED);
    }
    if output == 0 {
        return sce_kernel_error(COMMON_DIALOG_ERROR_ARG_NULL);
    }
    write_guest_u32(output, 0);
    write_guest_u32(output + 4, MESSAGE_DIALOG_RESULT_USER_CANCELED);
    write_guest_u32(output + 8, MESSAGE_DIALOG_BUTTON_INVALID);
    0
}

const COREDUMP_ERROR_ALREADY_REGISTERED: u32 = 0x8002_0011;
const COREDUMP_ERROR_PARAM: u32 = 0x8002_0016;
/// Largest handler stack the console accepts.
const COREDUMP_MAXIMUM_STACK_SIZE: u64 = 0x2000_0000;

/// Records the guest's coredump handler so a later fault path can run it.
fn coredump_register_handler(arguments: [u64; 6], state: &mut ProbeDispatchState) -> u64 {
    let (handler, stack_size) = (arguments[0], arguments[1]);
    if state.coredump_handler.is_some() {
        return sce_kernel_error(COREDUMP_ERROR_ALREADY_REGISTERED);
    }
    if handler == 0 || stack_size >= COREDUMP_MAXIMUM_STACK_SIZE {
        return sce_kernel_error(COREDUMP_ERROR_PARAM);
    }
    state.coredump_handler = Some((handler, stack_size));
    0
}

/// Number of microseconds in the RTC tick epoch before the Unix epoch.
const RTC_UNIX_EPOCH_TICK_MICROSECONDS: u64 = 62_135_596_800_000_000;
const RTC_ERROR_INVALID_POINTER: u32 = 0x80b5_0002;
const RTC_ERROR_INVALID_ARGUMENT: u32 = 0x80b5_0001;

/// Fills an `SceRtcDateTime` with the current wall-clock time in the host's local zone.
///
/// The layout is six 16-bit fields followed by the microsecond remainder. Leaving it untouched is
/// not a benign failure: engines validate the decomposed date and abort the process when every
/// field reads zero, because year zero is not a representable date.
fn rtc_get_current_clock_local_time(output: u64) -> u64 {
    if output == 0 {
        return sce_kernel_error(RTC_ERROR_INVALID_POINTER);
    }
    let now = std::time::SystemTime::now();
    let (unix_seconds, microsecond) = match now.duration_since(std::time::UNIX_EPOCH) {
        Ok(duration) => (
            i64::try_from(duration.as_secs()).unwrap_or(0),
            duration.subsec_micros(),
        ),
        Err(error) => (-i64::try_from(error.duration().as_secs()).unwrap_or(0), 0),
    };
    let (offset_seconds, _) = guest_local_timezone(unix_seconds);
    let local_seconds = unix_seconds.saturating_add(offset_seconds);
    let (year, month, day, hour, minute, second) = civil_from_unix_seconds(local_seconds);
    write_guest_u16(output, year);
    write_guest_u16(output + 2, month);
    write_guest_u16(output + 4, day);
    write_guest_u16(output + 6, hour);
    write_guest_u16(output + 8, minute);
    write_guest_u16(output + 10, second);
    write_guest_u32(output + 12, microsecond);
    0
}

/// Decomposes a Unix timestamp into a proleptic Gregorian civil date and time of day.
///
/// This is Howard Hinnant's `civil_from_days` algorithm: it shifts the era origin to 0000-03-01 so
/// the leap day falls at the end of a 400-year era, which keeps the arithmetic branch-free and
/// correct for dates before 1970.
fn civil_from_unix_seconds(seconds: i64) -> (u16, u16, u16, u16, u16, u16) {
    let days = seconds.div_euclid(86_400);
    let seconds_of_day = seconds.rem_euclid(86_400);
    let shifted = days + 719_468;
    let era = shifted.div_euclid(146_097);
    let day_of_era = shifted.rem_euclid(146_097);
    let year_of_era =
        (day_of_era - day_of_era / 1_460 + day_of_era / 36_524 - day_of_era / 146_096) / 365;
    let day_of_year = day_of_era - (365 * year_of_era + year_of_era / 4 - year_of_era / 100);
    let shifted_month = (5 * day_of_year + 2) / 153;
    let day = day_of_year - (153 * shifted_month + 2) / 5 + 1;
    // March is month 0 in the shifted calendar, so months 10 and 11 belong to the next year.
    let month = if shifted_month < 10 {
        shifted_month + 3
    } else {
        shifted_month - 9
    };
    let year = year_of_era + era * 400 + i64::from(month <= 2);
    (
        u16::try_from(year).unwrap_or(0),
        u16::try_from(month).unwrap_or(0),
        u16::try_from(day).unwrap_or(0),
        u16::try_from(seconds_of_day / 3_600).unwrap_or(0),
        u16::try_from(seconds_of_day / 60 % 60).unwrap_or(0),
        u16::try_from(seconds_of_day % 60).unwrap_or(0),
    )
}

fn ime_keyboard_get_resource_ids(arguments: [u64; 6]) -> u64 {
    let output = arguments[1];
    if output == 0 {
        return sce_kernel_error(SCE_KERNEL_ERROR_INVALID_ARGUMENT);
    }
    // The resource array contains the user id followed by five keyboard resource ids. REmu does
    // not expose a separate console keyboard device yet, so return an initialized empty list.
    guest_memset(output, 0, 24);
    write_guest_u32(output, low_u32(arguments[0]));
    0
}

fn ime_keyboard_get_info(output: u64) -> u64 {
    if output == 0 {
        return sce_kernel_error(SCE_KERNEL_ERROR_INVALID_ARGUMENT);
    }
    // SceImeKeyboardInfo is six 32-bit fields plus twelve reserved bytes.
    guest_memset(output, 0, 36);
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
const AGC_IT_SET_PREDICATION: u32 = 0x20;
const AGC_IT_COND_EXEC: u32 = 0x22;
const AGC_IT_INDEX_BASE: u32 = 0x26;
const AGC_IT_DRAW_INDEX_2: u32 = 0x27;
const AGC_IT_INDEX_TYPE: u32 = 0x2a;
const AGC_IT_DRAW_INDEX_AUTO: u32 = 0x2d;
const AGC_IT_NUM_INSTANCES: u32 = 0x2f;
const AGC_IT_DRAW_INDEX_OFFSET: u32 = 0x35;
const AGC_IT_WRITE_DATA: u32 = 0x37;
const AGC_IT_WAIT_REG_MEM: u32 = 0x3c;
const AGC_IT_INDIRECT_BUFFER: u32 = 0x3f;
const AGC_IT_EVENT_WRITE: u32 = 0x46;
const AGC_IT_RELEASE_MEM: u32 = 0x49;
const AGC_IT_DMA_DATA: u32 = 0x50;
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
const AGC_R_WRITE_DATA: u32 = 0x15;
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
const AGC_DB_DEPTH_SIZE_XY: u32 = 0x007;
const AGC_DB_Z_INFO: u32 = 0x010;
const AGC_DB_Z_READ_BASE: u32 = 0x012;
const AGC_DB_Z_WRITE_BASE: u32 = 0x014;
const AGC_DB_Z_READ_BASE_HI: u32 = 0x01a;
const AGC_DB_Z_WRITE_BASE_HI: u32 = 0x01c;
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

fn video_out_get_output_status(arguments: [u64; 6], state: &mut ProbeDispatchState) -> u64 {
    if low_i32(arguments[0]) != 1 || !state.video_out.is_open {
        return sce_kernel_error(VIDEO_OUT_ERROR_INVALID_HANDLE);
    }
    let output = arguments[1];
    if output == 0 {
        return sce_kernel_error(VIDEO_OUT_ERROR_INVALID_ADDRESS);
    }
    guest_memset(output, 0, 0x30);
    let resolution_class =
        u32::from(state.video_out.width >= 3840 || state.video_out.height >= 2160)
            .saturating_add(1);
    write_guest_u32(output, resolution_class);
    write_guest_u32(output + 4, 1);
    write_guest_u64(output + 8, 60);
    0
}

fn video_out_set_flip_rate(arguments: [u64; 6], state: &mut ProbeDispatchState) -> u64 {
    if low_i32(arguments[0]) != 1 || !state.video_out.is_open {
        return sce_kernel_error(VIDEO_OUT_ERROR_INVALID_HANDLE);
    }
    if low_i32(arguments[1]) < 0 || low_i32(arguments[1]) > 2 {
        return sce_kernel_error(VIDEO_OUT_ERROR_INVALID_VALUE);
    }
    0
}

fn video_out_is_flip_pending(arguments: [u64; 6], state: &mut ProbeDispatchState) -> u64 {
    if low_i32(arguments[0]) != 1 || !state.video_out.is_open {
        return sce_kernel_error(VIDEO_OUT_ERROR_INVALID_HANDLE);
    }
    0
}

fn video_out_submit_flip(arguments: [u64; 6], state: &mut ProbeDispatchState) -> u64 {
    if low_i32(arguments[0]) != 1 || !state.video_out.is_open {
        return sce_kernel_error(VIDEO_OUT_ERROR_INVALID_HANDLE);
    }
    let buffer_index = low_u32(arguments[1]);
    // UINT32_MAX requests the initial blank scanout before display buffers are registered.
    if buffer_index == u32::MAX {
        return 0;
    }
    let Ok(index) = usize::try_from(buffer_index) else {
        return sce_kernel_error(VIDEO_OUT_ERROR_INVALID_INDEX);
    };
    if index >= state.video_out.display_buffers.len() {
        return sce_kernel_error(VIDEO_OUT_ERROR_INVALID_INDEX);
    }
    queue_video_out_flip(state, 1, buffer_index, arguments[3]);
    0
}

fn ampr_command_buffer_constructor(arguments: [u64; 6], state: &mut ProbeDispatchState) -> u64 {
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

fn ampr_apr_command_buffer_constructor(arguments: [u64; 6], state: &mut ProbeDispatchState) -> u64 {
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

fn ampr_command_buffer_set_buffer(arguments: [u64; 6], state: &mut ProbeDispatchState) -> u64 {
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

/// Byte size of one recorded APR command record.
const AMPR_COMMAND_RECORD_SIZE: u64 = 0x30;
/// Record tag for an equeue completion signal.
const AMPR_COMMAND_WRITE_EVENT_QUEUE: u32 = 2;

/// Reports the capacity of the memory attached to a command buffer.
fn ampr_command_buffer_get_size(arguments: [u64; 6], state: &ProbeDispatchState) -> u64 {
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
fn ampr_command_buffer_get_current_offset(arguments: [u64; 6], state: &ProbeDispatchState) -> u64 {
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
const fn ampr_measure_command_size() -> u64 {
    AMPR_COMMAND_RECORD_SIZE
}

/// Records a command that signals a kernel event queue when the buffer's work completes.
///
/// REmu executes APR transfers as they are recorded, so the queued completion is delivered when
/// the buffer is submitted rather than by a hardware pipeline.
fn ampr_command_buffer_write_kernel_event_queue(
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
    0
}

/// Releases a command buffer's recorded state without freeing the guest's memory.
fn ampr_command_buffer_destructor(arguments: [u64; 6], state: &mut ProbeDispatchState) -> u64 {
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
fn ampr_command_buffer_clear_buffer(arguments: [u64; 6], state: &mut ProbeDispatchState) -> u64 {
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

fn ampr_command_buffer_reset(arguments: [u64; 6], state: &mut ProbeDispatchState) -> u64 {
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
    0
}

fn ampr_file_id(path: &str) -> u32 {
    path.as_bytes()
        .iter()
        .fold(2_166_136_261_u32, |hash, byte| {
            (hash ^ u32::from(*byte)).wrapping_mul(16_777_619)
        })
}

fn ampr_resolve_file_paths(arguments: [u64; 6], state: &mut ProbeDispatchState) -> u64 {
    let (path_list, count, ids, sizes, error_index) = (
        arguments[0],
        arguments[1],
        arguments[2],
        arguments[3],
        arguments[4],
    );
    if path_list == 0 || count == 0 || count > 1024 || sizes == 0 {
        return sce_kernel_error(SCE_KERNEL_ERROR_INVALID_ARGUMENT);
    }
    for index in 0..count {
        let candidate = read_guest_u64(path_list + index * 8);
        let path = read_guest_cstring(candidate).or_else(|| {
            (index == 0)
                .then(|| read_guest_cstring(path_list))
                .flatten()
        });
        let Some(guest_path) = path else {
            return sce_kernel_error(SCE_KERNEL_ERROR_MEMORY_FAULT);
        };
        let Some((host_path, _)) = resolve_probe_guest_path(state, &guest_path) else {
            if error_index != 0 {
                write_guest_u32(error_index, low_u32(index));
            }
            return sce_kernel_error(SCE_KERNEL_ERROR_NOT_FOUND);
        };
        let Ok(metadata) = host_path.metadata() else {
            if error_index != 0 {
                write_guest_u32(error_index, low_u32(index));
            }
            return sce_kernel_error(SCE_KERNEL_ERROR_NOT_FOUND);
        };
        let file_id = ampr_file_id(&guest_path);
        state.ampr_files.insert(file_id, host_path.clone());
        state.ampr_file_handles.remove(&file_id);
        if ids != 0 {
            write_guest_u32(ids + index * 4, file_id);
        }
        write_guest_u64(sizes + index * 8, metadata.len());
        eprintln!(
            "[REMu][AMPR] resolved guest={guest_path:?} host={:?} id={file_id:#x} size={:#x}",
            host_path.display(),
            metadata.len()
        );
    }
    0
}

fn ampr_command_buffer_read_file(
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
    let Some(host_path) = state.ampr_files.get(&file_id).cloned() else {
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
    };
    let Ok(capacity) = usize::try_from(size) else {
        return sce_kernel_error(SCE_KERNEL_ERROR_INVALID_ARGUMENT);
    };
    if !state.ampr_file_handles.contains_key(&file_id) {
        let file = match std::fs::File::open(&host_path) {
            Ok(file) => file,
            Err(error) => {
                eprintln!(
                    "[REMu][AMPR] read could not open id={file_id:#x} host={:?}: {error}",
                    host_path.display()
                );
                return sce_kernel_error(SCE_KERNEL_ERROR_NOT_FOUND);
            }
        };
        state.ampr_file_handles.insert(file_id, file);
    }
    let file = state
        .ampr_file_handles
        .get(&file_id)
        .expect("AMPR file handle was opened above");
    let bytes_read = if capacity == 0 {
        0
    } else {
        // SAFETY: AMPR receives a live guest mapping sized for the requested transfer. Reading
        // directly into it avoids an allocation and a second full copy for every asset chunk.
        let destination =
            unsafe { std::slice::from_raw_parts_mut(destination as *mut u8, capacity) };
        let mut total = 0_usize;
        while total < destination.len() {
            let Some(offset) = file_offset.checked_add(total as u64) else {
                return sce_kernel_error(SCE_KERNEL_ERROR_INVALID_ARGUMENT);
            };
            let read = match file.read_at(&mut destination[total..], offset) {
                Ok(read) => read,
                Err(error) => {
                    eprintln!(
                        "[REMu][AMPR] read failed id={file_id:#x} offset={offset:#x} \
remaining={:#x}: {error}",
                        destination.len() - total
                    );
                    return sce_kernel_error(SCE_KERNEL_ERROR_NOT_FOUND);
                }
            };
            if read == 0 {
                break;
            }
            total += read;
        }
        total
    };
    let Some(command) = state.ampr_command_buffers.get_mut(&command_buffer) else {
        return sce_kernel_error(SCE_KERNEL_ERROR_INVALID_ARGUMENT);
    };
    if command.write_offset > command.size || 0x30 > command.size - command.write_offset {
        return sce_kernel_error(SCE_KERNEL_ERROR_INVALID_ARGUMENT);
    }
    let record = command.buffer + command.write_offset;
    guest_memset(record, 0, 0x30);
    write_guest_u32(record, 1);
    write_guest_u32(record + 4, file_id);
    write_guest_u64(record + 8, destination);
    write_guest_u64(record + 0x10, size);
    write_guest_u64(record + 0x18, file_offset);
    write_guest_u64(record + 0x20, u64::try_from(bytes_read).unwrap_or(u64::MAX));
    command.write_offset += 0x30;
    command.command_count += 1;
    if trace_ampr_reads() {
        eprintln!(
            "[REMu][AMPR] read id={file_id:#x} offset={file_offset:#x} requested={size:#x} read={:#x} destination={destination:#x}",
            bytes_read
        );
    }
    0
}

fn ampr_submit_command_buffer(arguments: [u64; 6], state: &mut ProbeDispatchState) -> u64 {
    if arguments[0] == 0 || !state.ampr_command_buffers.contains_key(&arguments[0]) {
        return sce_kernel_error(SCE_KERNEL_ERROR_INVALID_ARGUMENT);
    }
    state.next_apr_submission_id = state.next_apr_submission_id.wrapping_add(1).max(1);
    let id = state.next_apr_submission_id;
    state.apr_submissions.insert(id);
    // The buffer's transfers were performed as they were recorded, so its completions become
    // observable now, in recorded order.
    let completions = state
        .ampr_command_buffers
        .get_mut(&arguments[0])
        .map(|command| std::mem::take(&mut command.pending_completions))
        .unwrap_or_default();
    let announced = !completions.is_empty();
    for completion in completions {
        queue_kernel_event(
            state,
            KernelQueuedEvent {
                equeue: completion.equeue,
                ident: completion.ident,
                filter: KERNEL_EVENT_FILTER_AMPR,
                flags: 0,
                fflags: 1,
                data: 0,
                user_data: completion.user_data,
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

fn ampr_wait_command_buffer(arguments: [u64; 6], state: &mut ProbeDispatchState) -> u64 {
    let id = low_u32(arguments[0]);
    if state.apr_submissions.remove(&id) {
        0
    } else {
        sce_kernel_error(SCE_KERNEL_ERROR_NOT_FOUND)
    }
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
    // Gen5 takes the command buffer, index address, and element capacity. The index address is
    // both the CPU-readable allocation and the GPU virtual address; keeping the identity mapping
    // preserves the submission decoder's existing resource provenance without inventing a fourth
    // ABI argument.
    let index_buffer_address = arguments[1];
    let index_count = low_u32(arguments[2]);
    state
        .agc_index_resources
        .insert(index_buffer_address, index_buffer_address);
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
    let registrations = state
        .video_out
        .flip_events
        .iter()
        .map(|(&equeue, &user_data)| (equeue, user_data))
        .collect::<Vec<_>>();
    for (equeue, user_data) in registrations {
        queue_kernel_event(
            state,
            KernelQueuedEvent {
                equeue,
                ident: 6,
                filter: -13,
                flags: 0,
                fflags: 0,
                data: event_data,
                user_data,
            },
        );
        if trace_logarithmic_sequence(state.agc_submission_count, 8) {
            eprintln!(
                "[REMu][VIDEO] queued flip handle={handle} index={buffer_index} address={display_address:#x} equeue={equeue:#x}"
            );
        }
    }
    notify_kernel_equeue_waiters();
}

fn queue_agc_event_write(state: &mut ProbeDispatchState, event_type: u32) {
    let registrations = state
        .kernel_event_registrations
        .iter()
        .filter_map(|(&(equeue, filter, ident), &user_data)| {
            (filter == KERNEL_EVENT_FILTER_GRAPHICS).then_some((equeue, ident, filter, user_data))
        })
        .collect::<Vec<_>>();
    for (equeue, ident, filter, user_data) in registrations {
        queue_kernel_event(
            state,
            KernelQueuedEvent {
                equeue,
                ident,
                filter,
                flags: 0,
                fflags: 1,
                data: u64::from(event_type),
                user_data,
            },
        );
    }
    notify_kernel_equeue_waiters();
}

fn request_agc_submission_completion(state: &mut ProbeDispatchState) {
    let sequence = state.agc_submission_count;
    state.pending_agc_completions.push_back(sequence);
    remu_gpu::emit_event(remu_gpu::GpuEvent::SubmissionBoundary(sequence));
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct AgcVideoOutTarget {
    address: u64,
    attributes: VideoOutBufferAttributes,
}

fn agc_video_out_target(
    video_out: &VideoOutState,
    handle: u32,
    buffer_index: u32,
) -> Option<AgcVideoOutTarget> {
    if handle != 1 || !video_out.is_open {
        return None;
    }
    let address = *video_out.display_buffers.get(buffer_index as usize)?;
    let attributes = video_out
        .display_buffer_attributes
        .get(buffer_index as usize)
        .copied()
        .flatten()?;
    Some(AgcVideoOutTarget {
        address,
        attributes,
    })
}

fn agc_video_out_target_for_address(
    video_out: &VideoOutState,
    address: u64,
) -> Option<AgcVideoOutTarget> {
    let index = video_out
        .display_buffers
        .iter()
        .position(|&candidate| candidate == address)?;
    let attributes = video_out
        .display_buffer_attributes
        .get(index)
        .copied()
        .flatten()?;
    Some(AgcVideoOutTarget {
        address,
        attributes,
    })
}

fn agc_submit_ordered_video_out_flip(
    video_out: &VideoOutState,
    handle: u32,
    buffer_index: u32,
    flip_argument: u64,
) -> bool {
    if handle != 1 || !video_out.is_open {
        eprintln!("[REMu][VIDEO] ignored flip for invalid handle {handle}");
        return false;
    }
    if video_out
        .display_buffers
        .get(buffer_index as usize)
        .is_none()
    {
        eprintln!("[REMu][VIDEO] ignored flip for unregistered buffer index {buffer_index}");
        return false;
    }
    let Some(target) = agc_video_out_target(video_out, handle, buffer_index) else {
        eprintln!("[REMu][VIDEO] ignored flip without attributes for buffer index {buffer_index}");
        return false;
    };
    remu_gpu::emit_event(remu_gpu::GpuEvent::Flip {
        address: target.address,
        buffer_index,
        flip_argument,
        width: target.attributes.width,
        height: target.attributes.height,
        pitch: target.attributes.pitch,
        pixel_format: target.attributes.pixel_format,
        tiling_mode: target.attributes.tiling_mode,
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
        if trace_agc_registers() && register == AGC_VGT_PRIMITIVE_TYPE {
            eprintln!(
                "[REMu][AGC][REG] primitive={value:#x} source=indexed packet={packet_address:#x}"
            );
        }
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
            let register = first_register + index;
            let value = read_guest_u32(packet_address + u64::from(index + 2) * 4);
            if trace_agc_registers()
                && opcode == AGC_IT_SET_UCONFIG_REG
                && register == AGC_VGT_PRIMITIVE_TYPE
            {
                eprintln!(
                    "[REMu][AGC][REG] primitive={value:#x} source=direct packet={packet_address:#x}"
                );
            }
            destination.insert(register, value);
        }
        if trace_agc_registers()
            && opcode == AGC_IT_SET_SH_REG
            && first_register <= 0x013
            && first_register.saturating_add(length - 2) > 0x00c
        {
            let pixel_user_data = std::array::from_fn::<_, 8, _>(|index| {
                graphics
                    .shader_registers
                    .get(&(0x00c + u32::try_from(index).expect("pixel scalar index fits")))
                    .copied()
            });
            eprintln!(
                "[REMu][AGC][REG] pixel user-data direct packet={packet_address:#x} first={first_register:#x} count={} values={pixel_user_data:x?}",
                length - 2,
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
        if trace_agc_registers() && offset == AGC_VGT_PRIMITIVE_TYPE {
            eprintln!(
                "[REMu][AGC][REG] primitive={value:#x} source=indirect opcode={opcode:#x} table={registers_address:#x}"
            );
        }
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
    Disabled,
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
        if target_mask.is_some() {
            AgcColorTargetState::Disabled
        } else {
            AgcColorTargetState::Absent
        }
    } else {
        AgcColorTargetState::Programmed(targets)
    }
}

fn agc_has_active_depth_state(registers: &BTreeMap<u32, u32>) -> bool {
    let enabled = registers
        .get(&AGC_DB_DEPTH_CONTROL)
        .is_some_and(|control| control & 0x6 != 0)
        || registers
            .get(&AGC_DB_RENDER_CONTROL)
            .is_some_and(|control| control & 1 != 0);
    if !enabled
        || registers
            .get(&AGC_DB_Z_INFO)
            .is_none_or(|info| info.trailing_zeros() >= 2)
        || !registers.contains_key(&AGC_DB_DEPTH_SIZE_XY)
    {
        return false;
    }
    let read_address = (u64::from(
        registers
            .get(&AGC_DB_Z_READ_BASE_HI)
            .copied()
            .unwrap_or_default()
            & 0xff,
    ) << 40)
        | (u64::from(
            registers
                .get(&AGC_DB_Z_READ_BASE)
                .copied()
                .unwrap_or_default(),
        ) << 8);
    let write_address = (u64::from(
        registers
            .get(&AGC_DB_Z_WRITE_BASE_HI)
            .copied()
            .unwrap_or_default()
            & 0xff,
    ) << 40)
        | (u64::from(
            registers
                .get(&AGC_DB_Z_WRITE_BASE)
                .copied()
                .unwrap_or_default(),
        ) << 8);
    read_address != 0 || write_address != 0
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
            let Some(end) = address.checked_add(length) else {
                return false;
            };
            if cached_kernel_memory_readable_range_end(address)
                .is_some_and(|mapped_end| end <= mapped_end)
            {
                // SAFETY: guest mapping mutations are serialized by the process dispatch lock.
                // Shader evaluation runs while its graphics import retains that lock, and the
                // generation-checked range snapshot proves this identity-mapped span readable.
                unsafe {
                    std::ptr::copy_nonoverlapping(
                        address as *const u8,
                        output.as_mut_ptr(),
                        output.len(),
                    );
                }
                return true;
            }
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

    fn readable_prefix_length(&self, address: u64, maximum_length: usize) -> Option<usize> {
        if maximum_length == 0 {
            return Some(0);
        }
        #[cfg(target_os = "macos")]
        {
            let mapped_end = cached_kernel_memory_readable_range_end(address)?;
            usize::try_from(mapped_end.saturating_sub(address))
                .ok()
                .map(|available| available.min(maximum_length))
        }
        #[cfg(not(target_os = "macos"))]
        {
            let _ = (address, maximum_length);
            None
        }
    }
}

pub(crate) fn try_read_process_guest_u32(address: u64) -> Option<u32> {
    let mut bytes = [0_u8; 4];
    ProcessGuestMemory
        .read(address, &mut bytes)
        .then(|| u32::from_le_bytes(bytes))
}

pub(crate) fn process_guest_range_is_readable(address: u64, length: usize) -> bool {
    let mut bytes = vec![0_u8; length];
    ProcessGuestMemory.read(address, &mut bytes)
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
        .flatten()
        .map(Arc::new);
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
    #[error("partial-flush EVENT_WRITE packet has {actual} dwords, expected 2")]
    EventWritePacketLength { actual: u32 },
    #[error("partial-flush EVENT_WRITE control has unsupported bits 0x{0:08X}")]
    UnsupportedEventWriteControl(u32),
    #[error(
        "partial-flush EVENT_WRITE type 0x{event_type:02X} uses event index {actual}, expected 4"
    )]
    PartialFlushEventIndex { event_type: u32, actual: u32 },
    #[error(
        "EVENT_WRITE type 0x{event_type:02X} index {event_index} with {length} dwords is not implemented"
    )]
    UnsupportedEventWrite {
        event_type: u32,
        event_index: u32,
        length: u32,
    },
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
    let shader_evaluation = evaluate_shader_register_snapshot_cached(
        &shader_registers,
        &shader,
        AGC_COMPUTE_USER_DATA_0,
        0,
        Some(compute_system_registers),
        remu_shader::EvaluationOptions {
            capture_global_binding_data: false,
            ..remu_shader::EvaluationOptions::default()
        },
        &mut graphics.shader_evaluation_cache,
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
        shader_evaluation: Some(shader_evaluation),
        shader_registers,
    })
}

fn agc_prepare_draw_color_targets(
    graphics: &mut AgcGraphicsState,
) -> (Vec<remu_gpu::GpuColorTarget>, bool) {
    let (targets, absent) = match agc_color_target_state(&graphics.context_registers) {
        AgcColorTargetState::Absent => (Vec::new(), true),
        AgcColorTargetState::Disabled => (Vec::new(), false),
        AgcColorTargetState::Programmed(targets) => (targets, false),
        AgcColorTargetState::Malformed { slot } => {
            if trace_logarithmic_sequence(graphics.emitted_draws, 8) {
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

fn agc_draw_writes_image(draw: &remu_gpu::IndexedDraw) -> bool {
    [
        draw.export_shader_evaluation.as_deref(),
        draw.pixel_shader_evaluation.as_deref(),
    ]
    .into_iter()
    .flatten()
    .flat_map(|evaluation| &evaluation.image_bindings)
    .any(|binding| {
        binding.opcode.starts_with("ImageStore") || binding.opcode.starts_with("ImageAtomic")
    })
}

fn agc_shader_has_memory_side_effects(shader: Option<&remu_gpu::ShaderSnapshot>) -> bool {
    shader.is_some_and(|shader| {
        shader.program.instructions.iter().any(|instruction| {
            instruction.opcode.contains("Store") || instruction.opcode.contains("Atomic")
        })
    })
}

fn agc_draw_resources_evaluated(draw: &remu_gpu::IndexedDraw) -> bool {
    (draw.export_shader.is_none() || draw.export_shader_evaluation.is_some())
        && (draw.pixel_shader.is_none() || draw.pixel_shader_evaluation.is_some())
}

fn agc_image_binding_address(binding: &remu_shader::ImageBinding) -> Option<u64> {
    let low = u64::from(*binding.resource_descriptor.first()?);
    let high = u64::from(*binding.resource_descriptor.get(1)? & 0xff);
    Some(((high << 32) | low) << 8)
}

fn agc_is_large_linear_video_binding(binding: &remu_shader::ImageBinding) -> bool {
    let words = binding.resource_descriptor.as_slice();
    if words.len() < 4 {
        return false;
    }
    let width = ((words[1] >> 30) | ((words[2] & 0x3fff) << 2)).saturating_add(1);
    let height = ((words[2] >> 14) & 0xffff).saturating_add(1);
    let tile_mode = (words[3] >> 20) & 0x1f;
    let unified_format = (words[1] >> 20) & 0x1ff;
    let data_format = remu_shader::decode_unified_format(unified_format)
        .map_or(u32::MAX, |(data_format, _)| data_format);
    tile_mode == 0
        && width >= 700
        && height >= 500
        && matches!(data_format, 1 | 3)
        && (binding.opcode.starts_with("ImageSample") || binding.opcode.starts_with("ImageGather"))
}

fn agc_resolve_targetless_scanout(
    graphics: &mut AgcGraphicsState,
    draw: &mut remu_gpu::IndexedDraw,
    color_target_absent: bool,
    future_target: Option<AgcVideoOutTarget>,
) {
    if !color_target_absent
        || agc_has_active_depth_state(&graphics.context_registers)
        || !agc_draw_resources_evaluated(draw)
        || agc_draw_writes_image(draw)
    {
        return;
    }
    let Some(target) = future_target else {
        return;
    };
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
    let [export_slot] = export_slots.as_slice() else {
        if trace_logarithmic_sequence(draw.sequence, 8) {
            eprintln!(
                "[REMu][AGC] did not associate targetless draw={}: future flip target {:#x} has ambiguous pixel exports {export_slots:?}",
                draw.sequence, target.address,
            );
        }
        return;
    };
    let color_surface = graphics
        .known_color_targets
        .get(&target.address)
        .copied()
        .filter(|surface| {
            ((surface.attrib2 >> 14) & 0x3fff) + 1 == target.attributes.width
                && (surface.attrib2 & 0x3fff) + 1 == target.attributes.height
        });
    draw.scanout_target = Some(remu_gpu::GpuScanoutTarget {
        address: target.address,
        width: target.attributes.width,
        height: target.attributes.height,
        pitch: target.attributes.pitch,
        pixel_format: target.attributes.pixel_format,
        tiling_mode: target.attributes.tiling_mode,
        export_slot: *export_slot,
        write_mask: (export_masks >> (*export_slot * 4)) & 0xf,
        color_surface,
    });
    if trace_logarithmic_sequence(draw.sequence, 8) {
        eprintln!(
            "[REMu][AGC] resolved targetless draw={} against future scanout target={:#x} programmed_surface={}",
            draw.sequence,
            target.address,
            color_surface.is_some(),
        );
    }
}

fn agc_trace_draw_summary(draw: &remu_gpu::IndexedDraw) {
    if trace_agc_draws() || trace_logarithmic_sequence(draw.sequence, 8) {
        let pixel_user_data = std::array::from_fn::<_, 8, _>(|index| {
            draw.shader_registers
                .iter()
                .find(|register| {
                    register.offset
                        == 0x00c + u32::try_from(index).expect("pixel scalar index fits")
                })
                .map(|register| register.value)
        });
        let viewport = [0x10f, 0x110, 0x111, 0x112].map(|offset| {
            draw.context_registers
                .iter()
                .find(|register| register.offset == offset)
                .map(|register| register.value)
        });
        eprintln!(
            "[REMu][AGC] draw={} indexed={} count={} instances={} offset={} index_buffer={:#x}/{} resource={:#x} index_size={} primitive={:#x} es={:#x} ps={:#x} target={:#x} scanout={:#x} target_mask={:#x} viewport={viewport:x?} ps_user={pixel_user_data:x?}",
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
            draw.scanout_target.map_or(0, |target| target.address),
            draw.context_registers
                .iter()
                .find(|register| register.offset == AGC_CB_TARGET_MASK)
                .map_or(0, |register| register.value),
        );
    }
}

fn agc_emit_draw(
    graphics: &mut AgcGraphicsState,
    shaders: &mut AgcShaderRegistry,
    video_out: &VideoOutState,
    indexed: bool,
    element_offset: u32,
    element_count: u32,
    draw_flags: u32,
    future_scanout_target: Option<AgcVideoOutTarget>,
) {
    if element_count == 0 {
        return;
    }
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
    // A zero CB_TARGET_MASK explicitly disables every color write. With no active depth surface
    // or shader memory side effect, the hardware draw is observably a no-op. Unity emits many of
    // these while preparing command buffers; evaluating both shaders and sending them through the
    // Metal presenter consumed substantial CPU despite every pipeline being rejected there.
    if graphics
        .context_registers
        .get(&AGC_CB_TARGET_MASK)
        .is_some_and(|&mask| mask == 0)
        && !agc_has_active_depth_state(&graphics.context_registers)
        && !agc_shader_has_memory_side_effects(export_shader.as_ref())
        && !agc_shader_has_memory_side_effects(pixel_shader.as_ref())
    {
        return;
    }
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
        index_snapshot: None,
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
    draw.index_snapshot = snapshot_draw_indices(&draw);
    let export_user_data_base = export_user_data_register(&draw);
    let required_vertex_record_count = draw_required_vertex_record_count(&draw);
    let export_request = draw.export_shader.as_ref().map(|snapshot| {
        shader_evaluation_worker_request(
            &graphics.shader_registers,
            snapshot,
            export_user_data_base,
            8,
            required_vertex_record_count,
        )
    });
    let pixel_request = draw.pixel_shader.as_ref().map(|snapshot| {
        shader_evaluation_worker_request(&graphics.shader_registers, snapshot, 0x00c, 0, None)
    });
    if graphics.shader_evaluation_workers.is_none() {
        match ShaderEvaluationWorkers::new() {
            Ok(workers) => graphics.shader_evaluation_workers = Some(workers),
            Err(error) => {
                eprintln!("[REMu][AGC] parallel shader evaluation unavailable: {error}");
            }
        }
    }
    let (export_evaluation, pixel_evaluation) =
        if let Some(workers) = graphics.shader_evaluation_workers.as_mut() {
            evaluate_draw_shader_requests_parallel(
                workers,
                &mut graphics.shader_evaluation_cache,
                export_request,
                pixel_request,
            )
        } else {
            let export = export_request.map(|request| {
                request.and_then(|request| {
                    graphics
                        .shader_evaluation_cache
                        .evaluate(&request.state, request.options)
                })
            });
            let pixel = pixel_request.map(|request| {
                request.and_then(|request| {
                    graphics
                        .shader_evaluation_cache
                        .evaluate(&request.state, request.options)
                })
            });
            (export, pixel)
        };
    if let Some(evaluation) = export_evaluation {
        draw.export_shader_evaluation = evaluation
            .map_err(|error| {
                if graphics
                    .failed_shader_evaluations
                    .insert((0, draw.export_shader_address))
                {
                    eprintln!(
                        "[REMu][AGC][SHADER] export evaluation failed at {:#x}: {error}",
                        draw.export_shader_address
                    );
                }
            })
            .ok();
    }
    if let Some(evaluation) = pixel_evaluation {
        draw.pixel_shader_evaluation = evaluation
            .map_err(|error| {
                if graphics
                    .failed_shader_evaluations
                    .insert((1, draw.pixel_shader_address))
                {
                    let control_flow = draw.pixel_shader.as_ref().map_or_else(
                        || "missing shader".to_owned(),
                        |snapshot| shader_control_flow_summary(&snapshot.program),
                    );
                    eprintln!(
                        "[REMu][AGC][SHADER] pixel evaluation failed at {:#x}: {error}; {}",
                        draw.pixel_shader_address, control_flow,
                    );
                }
            })
            .ok();
    }
    if trace_logarithmic_sequence(draw.sequence, 8) {
        let mut cache_stats = graphics.shader_evaluation_cache.stats();
        if let Some(workers) = graphics.shader_evaluation_workers.as_ref() {
            cache_stats = cache_stats.merge(workers.stats());
        }
        eprintln!(
            "[REMu][AGC] shader evaluation cache draw={} hits={} misses={} failures={} invalidations={} entries={} evictions={} pending_admissions={}",
            draw.sequence,
            cache_stats.hits,
            cache_stats.misses,
            cache_stats.failures,
            cache_stats.invalidations,
            cache_stats.entries,
            cache_stats.evictions,
            cache_stats.pending_admissions,
        );
    }
    agc_resolve_targetless_scanout(
        graphics,
        &mut draw,
        color_target_absent,
        future_scanout_target,
    );
    let samples_large_linear_video = draw
        .pixel_shader_evaluation
        .as_deref()
        .into_iter()
        .flat_map(|evaluation| &evaluation.image_bindings)
        .any(agc_is_large_linear_video_binding);
    if samples_large_linear_video {
        static LOGGED_VIDEO_DRAWS: OnceLock<Mutex<HashSet<u64>>> = OnceLock::new();
        let mut logged = LOGGED_VIDEO_DRAWS
            .get_or_init(|| Mutex::new(HashSet::new()))
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        if logged.len() < 16 && logged.insert(draw.pixel_shader_address) {
            let image_bindings = draw
                .pixel_shader_evaluation
                .as_deref()
                .into_iter()
                .flat_map(|evaluation| &evaluation.image_bindings)
                .map(|binding| {
                    (
                        binding.opcode,
                        agc_image_binding_address(binding),
                        binding.resource_descriptor.clone(),
                    )
                })
                .collect::<Vec<_>>();
            eprintln!(
                "[REMu][AGC][VIDEO] draw={} ps={:#x} color_absent={} target={:#x} target_mask={:?} shader_mask={:?} future={:#x} scanout={:#x} depth={} writes_image={} exports={:#x} bindings={image_bindings:x?}",
                draw.sequence,
                draw.pixel_shader_address,
                color_target_absent,
                draw.color_target_address,
                graphics.context_registers.get(&AGC_CB_TARGET_MASK),
                graphics.context_registers.get(&AGC_CB_SHADER_MASK),
                future_scanout_target.map_or(0, |target| target.address),
                draw.scanout_target.map_or(0, |target| target.address),
                agc_has_active_depth_state(&graphics.context_registers),
                agc_draw_writes_image(&draw),
                draw.pixel_shader
                    .as_ref()
                    .map_or(0, |shader| shader.program.pixel_color_export_masks()),
            );
        }
    }
    if trace_agc_shaders() {
        for address in [
            draw.export_shader_evaluation.as_deref(),
            draw.pixel_shader_evaluation.as_deref(),
        ]
        .into_iter()
        .flatten()
        .flat_map(|evaluation| &evaluation.image_bindings)
        .filter(|binding| {
            binding.opcode.starts_with("ImageSample") || binding.opcode.starts_with("ImageGather")
        })
        .filter_map(agc_image_binding_address)
        {
            graphics.sampled_texture_addresses.insert(address);
        }
        let pair = (draw.export_shader_address, draw.pixel_shader_address);
        let scanout_address = draw
            .scanout_target
            .map(|target| target.address)
            .or_else(|| {
                agc_video_out_target_for_address(video_out, draw.color_target_address)
                    .map(|target| target.address)
            })
            .or_else(|| {
                future_scanout_target
                    .filter(|target| target.address == draw.color_target_address)
                    .map(|target| target.address)
            });
        let explicit_target = trace_agc_target_address();
        let trace_pair = if let Some(address) = explicit_target {
            if address == draw.color_target_address
                && graphics.traced_target_shader_pairs.insert(pair)
            {
                eprintln!(
                    "[REMu][AGC][SHADER][TARGET] draw={} target={:#x} es={:#x} ps={:#x}",
                    draw.sequence,
                    draw.color_target_address,
                    draw.export_shader_address,
                    draw.pixel_shader_address,
                );
                true
            } else {
                false
            }
        } else if graphics
            .sampled_texture_addresses
            .contains(&draw.color_target_address)
        {
            if graphics.traced_sampled_target_shader_pairs.insert(pair) {
                eprintln!(
                    "[REMu][AGC][SHADER][SAMPLED-TARGET] draw={} target={:#x} es={:#x} ps={:#x}",
                    draw.sequence,
                    draw.color_target_address,
                    draw.export_shader_address,
                    draw.pixel_shader_address,
                );
                true
            } else {
                false
            }
        } else if let Some(address) = scanout_address {
            if graphics.traced_scanout_shader_pairs.insert(pair) {
                eprintln!(
                    "[REMu][AGC][SHADER][SCANOUT] draw={} target={address:#x} es={:#x} ps={:#x}",
                    draw.sequence, draw.export_shader_address, draw.pixel_shader_address,
                );
                true
            } else {
                false
            }
        } else if graphics.traced_shader_pairs.len() < 16 {
            graphics.traced_shader_pairs.insert(pair)
        } else {
            false
        };
        if trace_pair {
            trace_agc_shader_draw(&draw);
        }
    }
    agc_trace_draw_summary(&draw);
    remu_gpu::emit_event(remu_gpu::GpuEvent::IndexedDraw(Box::new(draw)));
}

fn shader_control_flow_summary(program: &remu_shader::Program) -> String {
    let branches = program
        .instructions
        .iter()
        .filter(|instruction| instruction.encoding == remu_shader::ShaderEncoding::Sopp)
        .filter(|instruction| {
            instruction.opcode.starts_with("SBranch") || instruction.opcode.starts_with("SCbranch")
        })
        .map(|instruction| {
            let encoded_offset = instruction.words.first().copied().unwrap_or_default() as u16;
            let offset = i16::from_ne_bytes(encoded_offset.to_ne_bytes());
            let next = i64::from(instruction.pc)
                + i64::try_from(instruction.words.len().saturating_mul(4)).unwrap_or(i64::MAX);
            let target = next.saturating_add(i64::from(offset) * 4);
            format!("{:#x}:{}->{target:#x}", instruction.pc, instruction.opcode)
        })
        .collect::<Vec<_>>();
    format!(
        "instructions={} branches=[{}]",
        program.instructions.len(),
        branches.join(",")
    )
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
        trace_shader_evaluation(
            draw,
            draw.export_shader.as_ref(),
            export_user_data_base,
            8,
            draw_required_vertex_record_count(draw),
        ),
        trace_shader_evaluation(draw, draw.pixel_shader.as_ref(), 0x00c, 0, None),
    ];
    let msl_summary = trace_msl_pair(draw, export_user_data_base);
    eprintln!(
        "[REMu][AGC][SHADER] es={:#x} ps={:#x} es_user_base={export_user_data_base:#x} indexed={} count={} instances={} target={:#x} es_ir={}/{} ps_ir={}/{} es_eval={} ps_eval={} msl={} es_ops=[{}] ps_ops=[{}] es_prefix={:x?} ps_prefix={:x?} cx={:x?} sh={:x?} uc={:x?}",
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

fn draw_required_vertex_record_count(draw: &remu_gpu::IndexedDraw) -> Option<u32> {
    if !draw.indexed {
        return Some(if draw.primitive_type == 0x11 {
            4
        } else {
            draw.index_count
        });
    }
    draw.index_snapshot
        .as_deref()
        .unwrap_or_default()
        .iter()
        .copied()
        .max()
        .and_then(|maximum| maximum.checked_add(1))
}

fn snapshot_draw_indices(draw: &remu_gpu::IndexedDraw) -> Option<Arc<[u32]>> {
    if !draw.indexed || draw.index_buffer_address < 0x10_000 || draw.index_count > 1_048_576 {
        return None;
    }
    let element_size = match draw.index_size {
        0 => 2_usize,
        1 => 4,
        2 => 1,
        _ => return None,
    };
    let first = usize::try_from(draw.index_offset)
        .ok()?
        .checked_mul(element_size)?;
    let byte_count = usize::try_from(draw.index_count)
        .ok()?
        .checked_mul(element_size)?;
    let address = draw
        .index_buffer_address
        .checked_add(u64::try_from(first).ok()?)?;
    let mut bytes = vec![0_u8; byte_count];
    if !ProcessGuestMemory.read(address, &mut bytes) {
        return None;
    }
    Some(
        bytes
            .chunks_exact(element_size)
            .map(|value| match element_size {
                1 => u32::from(value[0]),
                2 => u32::from(u16::from_le_bytes([value[0], value[1]])),
                _ => u32::from_le_bytes([value[0], value[1], value[2], value[3]]),
            })
            .collect::<Vec<_>>()
            .into(),
    )
}

fn evaluate_shader_snapshot(
    draw: &remu_gpu::IndexedDraw,
    snapshot: &remu_gpu::ShaderSnapshot,
    user_data_register: u32,
    scalar_register_base: u32,
    required_vertex_record_count: Option<u32>,
) -> Result<remu_shader::ShaderEvaluation, String> {
    evaluate_shader_register_snapshot_with_options(
        &draw.shader_registers,
        snapshot,
        user_data_register,
        scalar_register_base,
        None,
        remu_shader::EvaluationOptions {
            required_vertex_record_count,
            capture_global_binding_data: false,
            ..remu_shader::EvaluationOptions::default()
        },
    )
}

fn shader_evaluation_worker_request(
    shader_registers: &BTreeMap<u32, u32>,
    snapshot: &remu_gpu::ShaderSnapshot,
    user_data_register: u32,
    scalar_register_base: u32,
    required_vertex_record_count: Option<u32>,
) -> Result<ShaderEvaluationWorkerRequest, String> {
    Ok(ShaderEvaluationWorkerRequest {
        state: remu_shader::build_shader_state(
            Arc::clone(&snapshot.program),
            user_data_register,
            scalar_register_base,
            None,
            |offset| shader_registers.get(&offset).copied(),
        )
        .map_err(|error| format!("state-error({error})"))?,
        options: remu_shader::EvaluationOptions {
            required_vertex_record_count,
            capture_global_binding_data: false,
            ..remu_shader::EvaluationOptions::default()
        },
    })
}

fn evaluate_draw_shader_requests_parallel(
    workers: &mut ShaderEvaluationWorkers,
    local_cache: &mut ShaderEvaluationCache,
    export: Option<Result<ShaderEvaluationWorkerRequest, String>>,
    pixel: Option<Result<ShaderEvaluationWorkerRequest, String>>,
) -> (
    Option<Result<Arc<remu_shader::ShaderEvaluation>, String>>,
    Option<Result<Arc<remu_shader::ShaderEvaluation>, String>>,
) {
    let evaluate_local = |request: ShaderEvaluationWorkerRequest,
                          cache: &mut ShaderEvaluationCache| {
        cache.evaluate(&request.state, request.options)
    };
    match (export, pixel) {
        (Some(Ok(export)), Some(Ok(pixel))) => {
            // Keep the graphics task productive while the shorter shader runs on the helper. The
            // decoded instruction count is a stable cost proxy on cache misses. Running the larger
            // stage locally makes the helper likely to finish before this thread reaches the join,
            // while retaining two-core resource discovery and ordered draw emission.
            if export.state.program.instructions.len() >= pixel.state.program.instructions.len() {
                let pixel_submit = workers.parallel.submit(pixel);
                let export = evaluate_local(export, local_cache);
                let pixel = pixel_submit.and_then(|()| workers.parallel.receive());
                (Some(export), Some(pixel))
            } else {
                let export_submit = workers.parallel.submit(export);
                let pixel = evaluate_local(pixel, local_cache);
                let export = export_submit.and_then(|()| workers.parallel.receive());
                (Some(export), Some(pixel))
            }
        }
        (Some(Ok(export)), pixel) => (
            Some(evaluate_local(export, local_cache)),
            pixel.map(|result| {
                result.and_then(|_| unreachable!("pixel request was not successful"))
            }),
        ),
        (export, Some(Ok(pixel))) => (
            export.map(|result| {
                result.and_then(|_| unreachable!("export request was not successful"))
            }),
            Some(evaluate_local(pixel, local_cache)),
        ),
        (export, pixel) => (
            export.map(|result| {
                result.and_then(|_| unreachable!("export request was not successful"))
            }),
            pixel.map(|result| {
                result.and_then(|_| unreachable!("pixel request was not successful"))
            }),
        ),
    }
}

fn evaluate_shader_register_snapshot_cached(
    shader_registers: &[remu_gpu::GpuRegister],
    snapshot: &remu_gpu::ShaderSnapshot,
    user_data_register: u32,
    scalar_register_base: u32,
    compute_system_registers: Option<remu_shader::ComputeSystemRegisters>,
    options: remu_shader::EvaluationOptions,
    cache: &mut ShaderEvaluationCache,
) -> Result<Arc<remu_shader::ShaderEvaluation>, String> {
    let state = build_shader_register_state(
        shader_registers,
        snapshot,
        user_data_register,
        scalar_register_base,
        compute_system_registers,
    )?;
    cache.evaluate(&state, options)
}

fn evaluate_shader_register_snapshot_with_options(
    shader_registers: &[remu_gpu::GpuRegister],
    snapshot: &remu_gpu::ShaderSnapshot,
    user_data_register: u32,
    scalar_register_base: u32,
    compute_system_registers: Option<remu_shader::ComputeSystemRegisters>,
    options: remu_shader::EvaluationOptions,
) -> Result<remu_shader::ShaderEvaluation, String> {
    let state = build_shader_register_state(
        shader_registers,
        snapshot,
        user_data_register,
        scalar_register_base,
        compute_system_registers,
    )?;
    remu_shader::evaluate_shader(&ProcessGuestMemory, &state, options)
        .map_err(|error| format!("evaluation-error({error})"))
}

fn build_shader_register_state(
    shader_registers: &[remu_gpu::GpuRegister],
    snapshot: &remu_gpu::ShaderSnapshot,
    user_data_register: u32,
    scalar_register_base: u32,
    compute_system_registers: Option<remu_shader::ComputeSystemRegisters>,
) -> Result<remu_shader::ShaderState, String> {
    remu_shader::build_shader_state(
        Arc::clone(&snapshot.program),
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
    .map_err(|error| format!("state-error({error})"))
}

fn trace_msl_pair(draw: &remu_gpu::IndexedDraw, export_user_data_base: u32) -> String {
    let Some(export_snapshot) = draw.export_shader.as_ref() else {
        return "missing-es".to_owned();
    };
    let Some(pixel_snapshot) = draw.pixel_shader.as_ref() else {
        return "missing-ps".to_owned();
    };
    let export_evaluation = match evaluate_shader_snapshot(
        draw,
        export_snapshot,
        export_user_data_base,
        8,
        draw_required_vertex_record_count(draw),
    ) {
        Ok(evaluation) => evaluation,
        Err(error) => return format!("es-{error}"),
    };
    let pixel_evaluation = match evaluate_shader_snapshot(draw, pixel_snapshot, 0x00c, 0, None) {
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
    let Some(directory) = std::env::var_os("REMU_DUMP_SCANOUT_MSL_DIR")
        .or_else(|| std::env::var_os("REMU_DUMP_MSL_DIR"))
        .map(PathBuf::from)
    else {
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
    required_vertex_record_count: Option<u32>,
) -> String {
    let Some(snapshot) = snapshot else {
        return "missing".to_owned();
    };
    match evaluate_shader_snapshot(
        draw,
        snapshot,
        user_data_register,
        scalar_register_base,
        required_vertex_record_count,
    ) {
        Ok(evaluation) => {
            let images = evaluation
                .image_bindings
                .iter()
                .map(|binding| {
                    format!(
                        "{:#x}:{}:r={:x?}:s={:x?}:m={:?}",
                        binding.pc,
                        binding.opcode,
                        binding.resource_descriptor,
                        binding.sampler_descriptor,
                        binding.mip_level,
                    )
                })
                .collect::<Vec<_>>()
                .join(",");
            let globals = evaluation
                .global_memory_bindings
                .iter()
                .map(|binding| {
                    let prefix = binding
                        .data
                        .chunks_exact(4)
                        .take(4)
                        .map(|word| {
                            u32::from_le_bytes(word.try_into().expect("four-byte binding word"))
                        })
                        .collect::<Vec<_>>();
                    format!(
                        "s{}@{:#x}/{}:{prefix:x?}",
                        binding.scalar_address, binding.base_address, binding.byte_count
                    )
                })
                .collect::<Vec<_>>()
                .join(",");
            format!(
                "ok(i={},g={},v={},images=[{images}],globals=[{globals}])",
                evaluation.image_bindings.len(),
                evaluation.global_memory_bindings.len(),
                evaluation.vertex_inputs.len(),
            )
        }
        Err(error) => format!("error({error})"),
    }
}

fn agc_emit_indexed_draw(
    packet_address: u64,
    graphics: &mut AgcGraphicsState,
    shaders: &mut AgcShaderRegistry,
    video_out: &VideoOutState,
    future_scanout_target: Option<AgcVideoOutTarget>,
) {
    agc_emit_draw(
        graphics,
        shaders,
        video_out,
        true,
        read_guest_u32(packet_address + 8),
        read_guest_u32(packet_address + 12),
        read_guest_u32(packet_address + 16),
        future_scanout_target,
    );
}

#[derive(Clone, Copy)]
struct AgcSubmissionPacket {
    address: u64,
    length: u32,
    opcode: u32,
    register: u32,
    predicated: bool,
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
    if trace_agc_registers() {
        eprintln!(
            "[REMu][AGC][REG] clear state packet={packet_address:#x} control={control:#x} cx={} sh={} uc={}",
            graphics.context_registers.len(),
            graphics.shader_registers.len(),
            graphics.user_config_registers.len(),
        );
    }
    graphics.reset_pipeline();
    Ok(())
}

fn decode_agc_pipeline_barrier(
    packet_address: u64,
    length: u32,
) -> Result<remu_gpu::GpuPipelineBarrier, AgcSubmissionError> {
    let control = read_guest_u32(packet_address + 4);
    let event_type = control & 0x3f;
    let event_index = (control >> 8) & 0x7;
    let wait_for = match event_type {
        0x07 => remu_gpu::GpuPipelineStage::ComputeShader,
        0x0f => remu_gpu::GpuPipelineStage::VertexShader,
        0x10 => remu_gpu::GpuPipelineStage::PixelShader,
        _ => {
            return Err(AgcSubmissionError::UnsupportedEventWrite {
                event_type,
                event_index,
                length,
            });
        }
    };
    if length != 2 {
        return Err(AgcSubmissionError::EventWritePacketLength { actual: length });
    }
    if control & !0x0000_073f != 0 {
        return Err(AgcSubmissionError::UnsupportedEventWriteControl(control));
    }
    if event_index != 4 {
        return Err(AgcSubmissionError::PartialFlushEventIndex {
            event_type,
            actual: event_index,
        });
    }
    Ok(remu_gpu::GpuPipelineBarrier { wait_for })
}

fn decode_agc_event_write(
    packet_address: u64,
    length: u32,
) -> Result<Option<remu_gpu::GpuPipelineBarrier>, AgcSubmissionError> {
    let control = read_guest_u32(packet_address + 4);
    let event_type = control & 0x3f;
    if matches!(event_type, 0x07 | 0x0f | 0x10) {
        return decode_agc_pipeline_barrier(packet_address, length).map(Some);
    }
    if length != 2 {
        return Err(AgcSubmissionError::EventWritePacketLength { actual: length });
    }
    // Non-addressed Gen5 events use only EVENT_TYPE. They signal registered graphics queues in
    // submission order; EVENT_INDEX is reserved for the three partial-flush forms above.
    if control & !0x3f != 0 {
        return if (control >> 8) & 0x7 != 0 {
            Err(AgcSubmissionError::UnsupportedEventWrite {
                event_type,
                event_index: (control >> 8) & 0x7,
                length,
            })
        } else {
            Err(AgcSubmissionError::UnsupportedEventWriteControl(control))
        };
    }
    Ok(None)
}

fn decode_agc_memory_write(
    packet_address: u64,
    length: u32,
    standard_packet: bool,
) -> Option<remu_gpu::GpuMemoryWrite> {
    if length < 5 {
        return None;
    }
    let control = read_guest_u32(packet_address + 4);
    let (destination, increment_address) = if standard_packet {
        // GFX10 PKT3_WRITE_DATA: DST_SEL is 11:8 and ADDR_INCR=0 advances the address.
        ((control >> 8) & 0x0f, control & (1 << 16) == 0)
    } else {
        // sceAgc's NOP wrapper stores four byte-sized fields: destination, cache policy,
        // increment suppression and write confirmation.
        (control & 0xff, (control >> 16) & 0xff == 0)
    };
    if !matches!(destination, 1 | 2 | 4 | 5) {
        return None;
    }
    let address = read_guest_u64(packet_address + 8);
    if address == 0 || !address.is_multiple_of(4) {
        return None;
    }
    let values = (0..length - 4)
        .map(|index| read_guest_u32(packet_address + 16 + u64::from(index) * 4))
        .collect::<Vec<_>>();
    static WRITE_DATA_TRACE_COUNT: AtomicUsize = AtomicUsize::new(0);
    let sequence = WRITE_DATA_TRACE_COUNT.fetch_add(1, Ordering::Relaxed) + 1;
    if sequence <= 32 || sequence.is_power_of_two() {
        eprintln!(
            "[REMu][AGC][SYNC] write sequence={sequence} packet={packet_address:#x} standard={standard_packet} destination={destination} address={address:#x} dwords={} increment={increment_address} values={values:?}",
            values.len()
        );
    }
    Some(remu_gpu::GpuMemoryWrite {
        address,
        values,
        increment_address,
    })
}

fn decode_agc_dma_data(
    packet_address: u64,
    length: u32,
    standard_packet: bool,
) -> Option<remu_gpu::GpuMemoryCopy> {
    let (source_address, destination_address, byte_count, source_selection, destination_selection) =
        if standard_packet {
            if length < 7 {
                return None;
            }
            let control = read_guest_u32(packet_address + 4);
            let command = read_guest_u32(packet_address + 24);
            (
                read_guest_u64(packet_address + 8),
                read_guest_u64(packet_address + 16),
                u64::from(command & 0x03ff_ffff),
                (control >> 29) & 0x3,
                (control >> 20) & 0x3,
            )
        } else {
            if length < 8 {
                return None;
            }
            let control = read_guest_u32(packet_address + 4);
            (
                read_guest_u64(packet_address + 24),
                read_guest_u64(packet_address + 16),
                u64::from(read_guest_u32(packet_address + 12)),
                (control >> 16) & 0xff,
                control & 0xff,
            )
        };
    if source_address < 0x10_000
        || destination_address < 0x10_000
        || byte_count == 0
        || byte_count > 512 * 1024 * 1024
    {
        return None;
    }
    Some(remu_gpu::GpuMemoryCopy {
        source_address,
        destination_address,
        byte_count,
        source_selection,
        destination_selection,
        cpu_applied: false,
    })
}

fn mirror_agc_cpu_memory_copy(copy: &remu_gpu::GpuMemoryCopy) -> bool {
    const MAX_MIRRORED_COPY_BYTES: u64 = 1024 * 1024;
    if !matches!(copy.source_selection, 0 | 3)
        || !matches!(copy.destination_selection, 0 | 3)
        || copy.byte_count > MAX_MIRRORED_COPY_BYTES
    {
        return false;
    }
    let Ok(byte_count) = usize::try_from(copy.byte_count) else {
        return false;
    };
    if ProcessGuestMemory.readable_prefix_length(copy.source_address, byte_count)
        != Some(byte_count)
        || ProcessGuestMemory.readable_prefix_length(copy.destination_address, byte_count)
            != Some(byte_count)
    {
        return false;
    }
    // SAFETY: both complete ranges were validated against the runtime's live guest mappings.
    // `copy` provides memmove semantics. A destination protected by CpuWriteWatch is made
    // writable by its fault handler and dirtied exactly as a guest CPU write would be.
    unsafe {
        std::ptr::copy(
            copy.source_address as *const u8,
            copy.destination_address as *mut u8,
            byte_count,
        );
    }
    true
}

fn agc_handle_submission_packet(
    packet: AgcSubmissionPacket,
    graphics: &mut AgcGraphicsState,
    shaders: &mut AgcShaderRegistry,
    index_resources: &BTreeMap<u64, u64>,
    video_out: &VideoOutState,
    flips: &mut Vec<(u32, u32, u64)>,
    event_writes: &mut Vec<u32>,
    future_scanout_target: Option<AgcVideoOutTarget>,
) -> Result<(), AgcSubmissionError> {
    let AgcSubmissionPacket {
        address: packet_address,
        length,
        opcode,
        register,
        predicated: _,
    } = packet;
    if opcode == AGC_IT_CLEAR_STATE {
        agc_apply_clear_state(packet_address, length, graphics)?;
    }
    if opcode == AGC_IT_NOP && matches!(register, AGC_R_DRAW_RESET | AGC_R_ACB_RESET) && length >= 2
    {
        if trace_agc_registers() {
            eprintln!(
                "[REMu][AGC][REG] queue reset packet={packet_address:#x} register={register:#x} cx={} sh={} uc={}",
                graphics.context_registers.len(),
                graphics.shader_registers.len(),
                graphics.user_config_registers.len(),
            );
        }
        graphics.reset_pipeline();
    }
    agc_apply_submission_registers(packet_address, length, opcode, register, graphics)?;
    if opcode == AGC_IT_SET_BASE {
        agc_apply_set_base(packet_address, length, graphics)?;
    }
    if opcode == AGC_IT_EVENT_WRITE {
        if let Some(barrier) = decode_agc_event_write(packet_address, length)? {
            remu_gpu::pipeline_barrier(barrier);
        } else {
            event_writes.push(read_guest_u32(packet_address + 4) & 0x3f);
        }
    }
    if ((opcode == AGC_IT_WRITE_DATA) || (opcode == AGC_IT_NOP && register == AGC_R_WRITE_DATA))
        && let Some(write) =
            decode_agc_memory_write(packet_address, length, opcode == AGC_IT_WRITE_DATA)
    {
        remu_gpu::emit_event(remu_gpu::GpuEvent::MemoryWrite(write));
    }
    if (opcode == AGC_IT_DMA_DATA || (opcode == AGC_IT_NOP && register == AGC_R_DMA_DATA))
        && let Some(mut copy) =
            decode_agc_dma_data(packet_address, length, opcode == AGC_IT_DMA_DATA)
    {
        // Shader evaluation happens while the command buffer is parsed, before the asynchronous
        // Metal consumer reaches this event. Mirror bounded CPU-sourced uploads now so later
        // draws observe the descriptors and vertex data that DMA_DATA established. The ordered
        // event remains authoritative for copies whose source is produced by prior GPU work.
        copy.cpu_applied = mirror_agc_cpu_memory_copy(&copy);
        static DMA_TRACE_COUNT: AtomicUsize = AtomicUsize::new(0);
        let sequence = DMA_TRACE_COUNT.fetch_add(1, Ordering::Relaxed) + 1;
        if sequence <= 32
            || sequence.is_power_of_two()
            || graphics
                .sampled_texture_addresses
                .contains(&copy.destination_address)
        {
            eprintln!(
                "[REMu][AGC][DMA] sequence={sequence} packet={packet_address:#x} standard={} src_sel={} dst_sel={} source={:#x} destination={:#x} bytes={:#x}",
                opcode == AGC_IT_DMA_DATA,
                copy.source_selection,
                copy.destination_selection,
                copy.source_address,
                copy.destination_address,
                copy.byte_count,
            );
        }
        remu_gpu::emit_event(remu_gpu::GpuEvent::MemoryCopy(copy));
    }
    if opcode == AGC_IT_NOP && register == AGC_R_ACQUIRE_MEM && length >= 8 {
        remu_gpu::acquire_memory(decode_agc_memory_acquire(packet_address));
    }
    if opcode == AGC_IT_NOP
        && register == AGC_R_RELEASE_MEM
        && length >= 7
        && let Some(release) = decode_agc_memory_release(packet_address, false)
    {
        remu_gpu::emit_event(remu_gpu::GpuEvent::MemoryRelease(release));
    }
    if opcode == AGC_IT_RELEASE_MEM
        && length >= 8
        && let Some(release) = decode_agc_memory_release(packet_address, true)
    {
        remu_gpu::emit_event(remu_gpu::GpuEvent::MemoryRelease(release));
    }
    if opcode == AGC_IT_NOP
        && matches!(register, AGC_R_WAIT_MEM_32 | AGC_R_WAIT_MEM_64)
        && length >= if register == AGC_R_WAIT_MEM_64 { 9 } else { 6 }
        && let Some(wait) = decode_agc_memory_wait(packet_address, register, false)
    {
        remu_gpu::emit_event(remu_gpu::GpuEvent::MemoryWait(wait));
    }
    if opcode == AGC_IT_WAIT_REG_MEM
        && length >= 7
        && let Some(wait) = decode_agc_memory_wait(packet_address, register, true)
    {
        remu_gpu::emit_event(remu_gpu::GpuEvent::MemoryWait(wait));
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
    if opcode == AGC_IT_DRAW_INDEX_2 && length >= 6 {
        graphics.index_buffer_count = read_guest_u32(packet_address + 4);
        graphics.index_buffer_address = read_guest_u64(packet_address + 8);
        graphics.index_resource_address = index_resources
            .get(&graphics.index_buffer_address)
            .copied()
            .unwrap_or_default();
        agc_emit_draw(
            graphics,
            shaders,
            video_out,
            true,
            0,
            read_guest_u32(packet_address + 16),
            read_guest_u32(packet_address + 20),
            future_scanout_target,
        );
    }
    if opcode == AGC_IT_DRAW_INDEX_OFFSET && length >= 5 {
        agc_emit_indexed_draw(
            packet_address,
            graphics,
            shaders,
            video_out,
            future_scanout_target,
        );
    }
    if opcode == AGC_IT_DRAW_INDEX_AUTO && length >= 3 {
        agc_emit_draw(
            graphics,
            shaders,
            video_out,
            false,
            0,
            read_guest_u32(packet_address + 4),
            read_guest_u32(packet_address + 8),
            future_scanout_target,
        );
    }
    if opcode == AGC_IT_NOP && register == AGC_R_DRAW_INDEX_AUTO && length >= 2 {
        agc_emit_draw(
            graphics,
            shaders,
            video_out,
            false,
            0,
            read_guest_u32(packet_address + 4),
            0,
            future_scanout_target,
        );
    }
    if matches!(opcode, AGC_IT_DISPATCH_DIRECT | AGC_IT_DISPATCH_INDIRECT) {
        let dispatch =
            agc_decode_compute_dispatch(packet_address, length, opcode, graphics, shaders)
                .map_err(|source| AgcSubmissionError::Compute {
                    packet_address,
                    source,
                })?;
        if trace_logarithmic_sequence(dispatch.sequence, 8) {
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
        if trace_agc_shaders()
            && dispatch.sequence <= 4
            && let Some(shader) = dispatch.shader.as_ref()
        {
            let opcodes = shader
                .program
                .instructions
                .iter()
                .map(|instruction| instruction.opcode.to_string())
                .collect::<Vec<_>>()
                .join(",");
            eprintln!(
                "[REMu][AGC][SHADER][COMPUTE] cs={:#x} wave={} ir={} ops=[{opcodes}] instructions={:#?}",
                dispatch.shader_address,
                dispatch.launch.wave_lane_count,
                shader.program.instructions.len(),
                shader.program.instructions,
            );
        }
        if trace_agc_shaders()
            && let Some(evaluation) = dispatch.shader_evaluation.as_deref()
            && evaluation.image_bindings.iter().any(|binding| {
                agc_image_binding_address(binding)
                    .is_some_and(|address| graphics.sampled_texture_addresses.contains(&address))
            })
            && graphics
                .traced_sampled_compute_shaders
                .insert(dispatch.shader_address)
        {
            let images = evaluation
                .image_bindings
                .iter()
                .map(|binding| {
                    format!(
                        "{:#x}:{}@{:#x}:r={:x?}",
                        binding.pc,
                        binding.opcode,
                        agc_image_binding_address(binding).unwrap_or_default(),
                        binding.resource_descriptor,
                    )
                })
                .collect::<Vec<_>>()
                .join(",");
            eprintln!(
                "[REMu][AGC][SHADER][COMPUTE-SAMPLED] dispatch={} cs={:#x} images=[{images}]",
                dispatch.sequence, dispatch.shader_address,
            );
        }
        remu_gpu::emit_event(remu_gpu::GpuEvent::ComputeDispatch(Box::new(dispatch)));
    }
    if opcode == AGC_IT_NOP && register == AGC_R_FLIP && length >= 6 {
        let handle = read_guest_u32(packet_address + 4);
        let buffer_index = read_guest_u32(packet_address + 8);
        let flip_argument = u64::from(read_guest_u32(packet_address + 16))
            | (u64::from(read_guest_u32(packet_address + 20)) << 32);
        if agc_submit_ordered_video_out_flip(video_out, handle, buffer_index, flip_argument) {
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

fn decode_agc_memory_release(
    packet_address: u64,
    standard_packet: bool,
) -> Option<remu_gpu::GpuMemoryRelease> {
    static TRACE_COUNT: AtomicUsize = AtomicUsize::new(0);
    let control = read_guest_u32(packet_address + 8);
    let address = read_guest_u64(packet_address + 12);
    let data = read_guest_u64(packet_address + 20);
    let data_selection = if standard_packet {
        (control >> 29) & 0x7
    } else {
        (control >> 16) & 0xff
    };
    let value = match data_selection {
        1 if address.is_multiple_of(4) => {
            Some(remu_gpu::GpuMemoryReleaseValue::Immediate32(low_u32(data)))
        }
        2 if address.is_multiple_of(8) => Some(remu_gpu::GpuMemoryReleaseValue::Immediate64(data)),
        3 | 4 if address.is_multiple_of(8) => Some(remu_gpu::GpuMemoryReleaseValue::Timestamp),
        _ => None,
    };
    let trace = TRACE_COUNT.fetch_add(1, Ordering::Relaxed) + 1;
    if trace <= 96 || trace.is_power_of_two() {
        eprintln!(
            "[REMu][AGC][SYNC] release packet={packet_address:#x} standard={standard_packet} control={control:#010x} selection={data_selection} address={address:#x} data={data:#x} emitted={}",
            value.is_some() && address != 0
        );
    }
    Some(remu_gpu::GpuMemoryRelease {
        address,
        value: value?,
    })
    .filter(|_| address != 0)
}

fn decode_agc_memory_wait(
    packet_address: u64,
    register: u32,
    standard_packet: bool,
) -> Option<remu_gpu::GpuMemoryWait> {
    static TRACE_COUNT: AtomicUsize = AtomicUsize::new(0);
    let is_64_bit = !standard_packet && register == AGC_R_WAIT_MEM_64;
    let (address, reference, mask, control) = if standard_packet {
        (
            read_guest_u64(packet_address + 8),
            u64::from(read_guest_u32(packet_address + 16)),
            u64::from(read_guest_u32(packet_address + 20)),
            read_guest_u32(packet_address + 4),
        )
    } else if is_64_bit {
        (
            read_guest_u64(packet_address + 4),
            read_guest_u64(packet_address + 20),
            read_guest_u64(packet_address + 12),
            read_guest_u32(packet_address + 28),
        )
    } else {
        (
            read_guest_u64(packet_address + 4),
            u64::from(read_guest_u32(packet_address + 20)),
            u64::from(read_guest_u32(packet_address + 12)),
            read_guest_u32(packet_address + 16),
        )
    };
    let compare_function = control & 0x7;
    let alignment = if is_64_bit { 8 } else { 4 };
    let decoded = if address == 0
        || mask == 0
        || !address.is_multiple_of(alignment)
        || matches!(compare_function, 0 | 7)
    {
        None
    } else {
        Some(remu_gpu::GpuMemoryWait {
            address,
            reference,
            mask,
            compare_function,
            is_64_bit,
        })
    };
    let trace = TRACE_COUNT.fetch_add(1, Ordering::Relaxed) + 1;
    if trace <= 64 || trace.is_power_of_two() {
        let current = decoded.map(|wait| {
            if wait.is_64_bit {
                read_guest_u64(wait.address)
            } else {
                u64::from(read_guest_u32(wait.address))
            }
        });
        eprintln!(
            "[REMu][AGC][SYNC] wait packet={packet_address:#x} standard={standard_packet} bits={} control={control:#010x} compare={compare_function} address={address:#x} mask={mask:#x} reference={reference:#x} current={current:?} satisfied={:?}",
            if is_64_bit { 64 } else { 32 },
            current
                .zip(decoded)
                .map(|(value, wait)| wait.is_satisfied_by(value))
        );
    }
    decoded
}

struct AgcSubmissionContext<'a> {
    graphics: &'a mut AgcGraphicsState,
    shaders: &'a mut AgcShaderRegistry,
    index_resources: &'a BTreeMap<u64, u64>,
    video_out: &'a VideoOutState,
    submission_count: &'a mut u64,
    flips: &'a mut Vec<(u32, u32, u64)>,
    event_writes: &'a mut Vec<u32>,
}

fn agc_packet_video_out_target(
    packet: AgcSubmissionPacket,
    video_out: &VideoOutState,
) -> Option<AgcVideoOutTarget> {
    if packet.opcode != AGC_IT_NOP || packet.register != AGC_R_FLIP || packet.length < 6 {
        return None;
    }
    agc_video_out_target(
        video_out,
        read_guest_u32(packet.address + 4),
        read_guest_u32(packet.address + 8),
    )
}

fn agc_scanout_inference_boundary(packet: AgcSubmissionPacket) -> bool {
    packet.predicated
        || matches!(
            packet.opcode,
            AGC_IT_SET_PREDICATION | AGC_IT_COND_EXEC | AGC_IT_INDIRECT_BUFFER | AGC_IT_CLEAR_STATE
        )
        || (packet.opcode == AGC_IT_NOP
            && matches!(packet.register, AGC_R_DRAW_RESET | AGC_R_ACB_RESET))
}

fn agc_future_scanout_targets(
    packets: &[AgcSubmissionPacket],
    video_out: &VideoOutState,
) -> Vec<Option<AgcVideoOutTarget>> {
    let mut future = vec![None; packets.len()];
    let mut next = None;
    for (index, packet) in packets.iter().copied().enumerate().rev() {
        if agc_scanout_inference_boundary(packet) {
            next = None;
            continue;
        }
        future[index] = next;
        if let Some(target) = agc_packet_video_out_target(packet, video_out) {
            next = Some(target);
        }
    }
    future
}

fn agc_parse_submission(
    command_address: u64,
    dword_count: u32,
    context: &mut AgcSubmissionContext<'_>,
) -> Result<(), AgcSubmissionError> {
    let mut offset = 0_u32;
    let mut packet_counts = BTreeMap::<(u32, u32), u32>::new();
    let mut packets = Vec::new();
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
        *packet_counts.entry((opcode, register)).or_default() += 1;
        packets.push(AgcSubmissionPacket {
            address: packet_address,
            length,
            opcode,
            register,
            predicated: header & 1 != 0,
        });
        offset += length;
    }

    // Validate newly decoded order-visible packet forms before emitting any GPU side effect from
    // the submission.
    for packet in &packets {
        if packet.opcode == AGC_IT_EVENT_WRITE {
            decode_agc_event_write(packet.address, packet.length)?;
        }
    }

    let future_scanout_targets = agc_future_scanout_targets(&packets, context.video_out);
    for (packet, future_scanout_target) in packets
        .iter()
        .copied()
        .zip(future_scanout_targets.into_iter())
    {
        agc_handle_submission_packet(
            packet,
            context.graphics,
            context.shaders,
            context.index_resources,
            context.video_out,
            context.flips,
            context.event_writes,
            future_scanout_target,
        )?;
    }

    *context.submission_count = context.submission_count.saturating_add(1);
    if trace_logarithmic_sequence(*context.submission_count, 8) {
        let packet_summary = packet_counts
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
    let mut event_writes = Vec::new();
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
            event_writes: &mut event_writes,
        },
    );
    for (handle, buffer_index, flip_argument) in flips {
        queue_video_out_flip(state, handle, buffer_index, flip_argument);
    }
    for event_type in event_writes {
        queue_agc_event_write(state, event_type);
    }
    match result {
        Ok(()) => {
            request_agc_submission_completion(state);
            0
        }
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
            let mut event_writes = Vec::new();
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
                    event_writes: &mut event_writes,
                },
            );
            for (handle, buffer_index, flip_argument) in flips {
                queue_video_out_flip(state, handle, buffer_index, flip_argument);
            }
            for event_type in event_writes {
                queue_agc_event_write(state, event_type);
            }
            match result {
                Ok(()) => request_agc_submission_completion(state),
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
    event_writes: &mut Vec<u32>,
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
            event_writes,
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
    let mut event_writes = Vec::new();
    let result = agc_submit_compute_commands(
        owner,
        command_address,
        dword_count,
        state,
        &mut flips,
        &mut event_writes,
    );
    for (handle, buffer_index, flip_argument) in flips {
        queue_video_out_flip(state, handle, buffer_index, flip_argument);
    }
    for event_type in event_writes {
        queue_agc_event_write(state, event_type);
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
        let mut event_writes = Vec::new();
        let result = agc_submit_compute_commands(
            owner,
            command_address,
            dword_count,
            state,
            &mut flips,
            &mut event_writes,
        );
        for (handle, buffer_index, flip_argument) in flips {
            queue_video_out_flip(state, handle, buffer_index, flip_argument);
        }
        for event_type in event_writes {
            queue_agc_event_write(state, event_type);
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
    if entry == 0 {
        return 0;
    }
    invoke_guest_entry_with_exit(entry, argument, 0)
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

fn pthread_attr_get_stack_address(arguments: [u64; 6], state: &ProbeDispatchState) -> u64 {
    if arguments[1] == 0 {
        return sce_kernel_error(SCE_KERNEL_ERROR_INVALID_ARGUMENT);
    }
    let Some(attributes) = pthread_attr_state(arguments[0], state) else {
        return sce_kernel_error(SCE_KERNEL_ERROR_INVALID_ARGUMENT);
    };
    write_guest_u64(arguments[1], attributes.stack_address);
    attributes.stack_address
}

fn pthread_attr_get_stack_size(arguments: [u64; 6], state: &ProbeDispatchState) -> u64 {
    if arguments[1] == 0 {
        return sce_kernel_error(SCE_KERNEL_ERROR_INVALID_ARGUMENT);
    }
    let Some(attributes) = pthread_attr_state(arguments[0], state) else {
        return sce_kernel_error(SCE_KERNEL_ERROR_INVALID_ARGUMENT);
    };
    write_guest_u64(arguments[1], attributes.stack_size);
    attributes.stack_size
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

fn pthread_attr_set_affinity(arguments: [u64; 6], state: &mut ProbeDispatchState) -> u64 {
    // Sony's scePthread form takes the 64-bit CPU mask directly as its second argument. This is
    // distinct from POSIX pthread_attr_setaffinity_np(attr, cpusetsize, cpuset), and interpreting
    // the mask as a byte count rejects the single-core masks used by FMOD's mixer threads.
    update_pthread_attr(arguments[0], state, |attributes| {
        attributes.affinity_mask = arguments[1];
    })
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

fn pthread_attr_get_schedule_param(arguments: [u64; 6], state: &ProbeDispatchState) -> u64 {
    if arguments[1] == 0 {
        return sce_kernel_error(SCE_KERNEL_ERROR_INVALID_ARGUMENT);
    }
    let Some(attributes) = pthread_attr_state(arguments[0], state) else {
        return sce_kernel_error(SCE_KERNEL_ERROR_INVALID_ARGUMENT);
    };
    write_guest_u32(arguments[1], attributes.schedule_priority.cast_unsigned());
    0
}

fn pthread_attr_set_inherit_schedule(arguments: [u64; 6], state: &mut ProbeDispatchState) -> u64 {
    let inherit_schedule = low_i32(arguments[1]);
    update_pthread_attr(arguments[0], state, |attributes| {
        attributes.inherit_schedule = inherit_schedule;
    })
}

fn pthread_attr_set_schedule_policy(arguments: [u64; 6], state: &mut ProbeDispatchState) -> u64 {
    let schedule_policy = low_i32(arguments[1]);
    update_pthread_attr(arguments[0], state, |attributes| {
        attributes.schedule_policy = schedule_policy;
    })
}

fn pthread_get_schedule_param(arguments: [u64; 6], state: &mut ProbeDispatchState) -> u64 {
    if arguments[0] == 0 || arguments[1] == 0 || arguments[2] == 0 {
        return sce_kernel_error(SCE_KERNEL_ERROR_INVALID_ARGUMENT);
    }
    let attributes = *state.thread_attributes.entry(arguments[0]).or_default();
    write_guest_u32(arguments[1], attributes.schedule_policy.cast_unsigned());
    write_guest_u32(arguments[2], attributes.schedule_priority.cast_unsigned());
    0
}

fn pthread_set_schedule_param(arguments: [u64; 6], state: &mut ProbeDispatchState) -> u64 {
    if arguments[0] == 0 || arguments[2] == 0 {
        return sce_kernel_error(SCE_KERNEL_ERROR_INVALID_ARGUMENT);
    }
    let policy = low_i32(arguments[1]);
    let priority = read_guest_u32(arguments[2]).cast_signed();
    let attributes = state.thread_attributes.entry(arguments[0]).or_default();
    attributes.schedule_policy = policy;
    attributes.schedule_priority = priority;
    0
}

fn pthread_set_affinity(arguments: [u64; 6], state: &mut ProbeDispatchState) -> u64 {
    if arguments[0] == 0 {
        return sce_kernel_error(SCE_KERNEL_ERROR_INVALID_ARGUMENT);
    }
    state
        .thread_attributes
        .entry(arguments[0])
        .or_default()
        .affinity_mask = arguments[1];
    0
}

fn pthread_set_priority(arguments: [u64; 6], state: &mut ProbeDispatchState) -> u64 {
    if arguments[0] == 0 {
        return sce_kernel_error(SCE_KERNEL_ERROR_INVALID_ARGUMENT);
    }
    state
        .thread_attributes
        .entry(arguments[0])
        .or_default()
        .schedule_priority = low_i32(arguments[1]);
    0
}

fn pthread_rename(arguments: [u64; 6], state: &mut ProbeDispatchState) -> u64 {
    if arguments[0] == 0 || arguments[1] == 0 {
        return sce_kernel_error(SCE_KERNEL_ERROR_INVALID_ARGUMENT);
    }
    if let Some(thread) = state.pthread_threads.get_mut(&arguments[0])
        && let Some(name) = read_guest_cstring(arguments[1])
    {
        thread.name = name;
    }
    0
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

fn pthread_mutex_attr_set_protocol(arguments: [u64; 6], state: &mut ProbeDispatchState) -> u64 {
    // Gen5 accepts NONE, INHERIT, and PROTECT. The host synchronization implementation does not
    // expose priority inheritance, but preserving the attribute is observable and must succeed.
    let protocol = match low_i32(arguments[1]) {
        value @ 0..=2 => value,
        _ => return sce_kernel_error(SCE_KERNEL_ERROR_INVALID_ARGUMENT),
    };
    let Some(key) = resolve_pthread_mutex_attr_key(arguments[0], state) else {
        return sce_kernel_error(SCE_KERNEL_ERROR_INVALID_ARGUMENT);
    };
    let object = state
        .pthread_mutex_attributes
        .get_mut(&key)
        .expect("the pthread mutex attribute key was just resolved");
    object.attributes.protocol = protocol;
    write_pthread_mutex_attr_object(object.handle, object.attributes);
    0
}

fn pthread_key_create(arguments: [u64; 6], state: &mut ProbeDispatchState) -> u64 {
    if arguments[0] == 0 {
        return u64::from(libc::EINVAL.cast_unsigned());
    }
    let Some(key) = state.next_pthread_key.checked_add(1).map(|next| {
        let key = state.next_pthread_key;
        state.next_pthread_key = next;
        key
    }) else {
        return u64::from(libc::EAGAIN.cast_unsigned());
    };
    state.pthread_keys.insert(key, arguments[1]);
    write_guest_u32(arguments[0], key);
    0
}

fn pthread_key_delete(key: u32, state: &mut ProbeDispatchState) -> u64 {
    if state.pthread_keys.remove(&key).is_none() {
        return u64::from(libc::ENOENT.cast_unsigned());
    }
    GUEST_PTHREAD_SPECIFIC.with_borrow_mut(|values| {
        values.remove(&key);
    });
    0
}

fn pthread_set_specific(arguments: [u64; 6], state: &ProbeDispatchState) -> u64 {
    let key = low_u32(arguments[0]);
    if !state.pthread_keys.contains_key(&key) {
        return u64::from(libc::ENOENT.cast_unsigned());
    }
    GUEST_PTHREAD_SPECIFIC.with_borrow_mut(|values| {
        values.insert(key, arguments[1]);
    });
    0
}

fn pthread_get_specific(key: u32) -> u64 {
    GUEST_PTHREAD_SPECIFIC.with_borrow(|values| values.get(&key).copied().unwrap_or(0))
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

fn np_universal_data_create_property_object(output: u64, state: &mut ProbeDispatchState) -> u64 {
    if output == 0 {
        return sce_kernel_error(np::UDS_ERROR_INVALID_ARGUMENT);
    }
    match state.np_universal_data.create_property_object() {
        Ok(object) => {
            write_guest_u64(output, object);
            0
        }
        Err(error) => sce_kernel_error(error),
    }
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

fn guest_net_errno_address() -> u64 {
    // libSceNet exposes a distinct thread-local errno cell. The guest TCB reservation is private
    // to each emulated pthread; libc errno occupies +0x40, so keep the network value beside it.
    let thread_pointer = current_guest_thread_pointer();
    if thread_pointer == 0 {
        0
    } else {
        thread_pointer.checked_add(0x44).unwrap_or(0)
    }
}

fn guest_net_unavailable(error: i32) -> u64 {
    let errno_address = guest_net_errno_address();
    if errno_address != 0 {
        write_guest_u32(errno_address, error.cast_unsigned());
    }
    // REmu runs dumped titles offline. Returning a real libSceNet error lets optional FMOD Live
    // Update/profiling sockets disable themselves without exposing a host listener.
    signed_i32_return(i32::from_ne_bytes(
        (0x8041_0100_u32 | error.cast_unsigned()).to_ne_bytes(),
    ))
}

fn guest_tls_get_addr(descriptor: u64) -> u64 {
    let thread_pointer = current_guest_thread_pointer();
    let Some(descriptor_offset) = descriptor.checked_add(8) else {
        return 0;
    };
    if descriptor == 0 || thread_pointer == 0 {
        return 0;
    }
    let module_id = read_guest_u64(descriptor);
    let offset = read_guest_u64(descriptor_offset);
    let Some(dtv_pointer_address) = thread_pointer.checked_add(8) else {
        return 0;
    };
    let dtv = read_guest_u64(dtv_pointer_address);
    let Some(maximum_module_address) = dtv.checked_add(8) else {
        return 0;
    };
    if dtv == 0 || module_id == 0 {
        return 0;
    }
    let maximum_module_id = read_guest_u64(maximum_module_address);
    if module_id > maximum_module_id {
        return 0;
    }
    let Some(entry_address) =
        tls_dtv_entry_offset(module_id).and_then(|entry_offset| dtv.checked_add(entry_offset))
    else {
        return 0;
    };
    let block = read_guest_u64(entry_address);
    if block == 0 {
        return 0;
    }
    block.checked_add(offset).unwrap_or(0)
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
fn allocate_main_direct_memory(arguments: [u64; 6], state: &mut ProbeDispatchState) -> u64 {
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
fn allocate_main_direct_memory(_arguments: [u64; 6], _state: &mut ProbeDispatchState) -> u64 {
    sce_kernel_error(SCE_KERNEL_ERROR_NOT_FOUND)
}

#[cfg(target_os = "macos")]
fn direct_memory_query(arguments: [u64; 6], state: &mut ProbeDispatchState) -> u64 {
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
fn direct_memory_query(_arguments: [u64; 6], _state: &mut ProbeDispatchState) -> u64 {
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
fn map_named_flexible_memory(arguments: [u64; 6], state: &mut ProbeDispatchState) -> u64 {
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
fn map_flexible_memory_core(
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
fn map_named_flexible_memory(_arguments: [u64; 6], _state: &mut ProbeDispatchState) -> u64 {
    sce_kernel_error(SCE_KERNEL_ERROR_NOT_FOUND)
}

/// Byte size of one `sceKernelBatchMap` entry: `start`, `offset`, `length`, `protection`, `type`,
/// two reserved bytes, and `operation`.
const BATCH_MAP_ENTRY_SIZE: u64 = 32;
const BATCH_MAP_OPERATION_MAP_DIRECT: u32 = 0;
const BATCH_MAP_OPERATION_UNMAP: u32 = 1;
const BATCH_MAP_OPERATION_PROTECT: u32 = 2;
const BATCH_MAP_OPERATION_MAP_FLEXIBLE: u32 = 3;
const BATCH_MAP_OPERATION_TYPE_PROTECT: u32 = 4;
/// `MAP_FIXED`. The single-argument batch entry point always maps at the caller's addresses.
const KERNEL_MAP_FIXED: u64 = 0x10;

#[cfg(target_os = "macos")]
fn kernel_batch_map(arguments: [u64; 6], state: &mut ProbeDispatchState) -> u64 {
    kernel_batch_map_core(arguments, KERNEL_MAP_FIXED, state)
}

#[cfg(target_os = "macos")]
fn kernel_batch_map_2(arguments: [u64; 6], state: &mut ProbeDispatchState) -> u64 {
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
fn kernel_batch_map(_arguments: [u64; 6], _state: &mut ProbeDispatchState) -> u64 {
    sce_kernel_error(SCE_KERNEL_ERROR_NOT_FOUND)
}

#[cfg(not(target_os = "macos"))]
fn kernel_batch_map_2(_arguments: [u64; 6], _state: &mut ProbeDispatchState) -> u64 {
    sce_kernel_error(SCE_KERNEL_ERROR_NOT_FOUND)
}

/// Reprotects a range and assigns it a new guest memory type in the same operation.
#[cfg(target_os = "macos")]
fn kernel_mtypeprotect(arguments: [u64; 6], state: &mut ProbeDispatchState) -> u64 {
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
    if let Err(error) = memory.protect_with_type(
        GuestAddress::new(aligned_address),
        aligned_end - aligned_address,
        protection,
        Some(memory_type),
    ) {
        return kernel_memory_error(&error);
    }
    refresh_kernel_memory_readable_ranges(memory);
    0
}

#[cfg(not(target_os = "macos"))]
fn kernel_mtypeprotect(_arguments: [u64; 6], _state: &mut ProbeDispatchState) -> u64 {
    sce_kernel_error(SCE_KERNEL_ERROR_NOT_FOUND)
}

#[cfg(target_os = "macos")]
fn kernel_reserve_virtual_range(arguments: [u64; 6], state: &mut ProbeDispatchState) -> u64 {
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
fn kernel_reserve_virtual_range(_arguments: [u64; 6], _state: &mut ProbeDispatchState) -> u64 {
    sce_kernel_error(SCE_KERNEL_ERROR_NOT_FOUND)
}

fn kernel_set_virtual_range_name(arguments: [u64; 6]) -> u64 {
    if arguments[0] == 0 || arguments[1] == 0 || arguments[2] == 0 {
        return sce_kernel_error(SCE_KERNEL_ERROR_INVALID_ARGUMENT);
    }
    match read_guest_cstring(arguments[2]) {
        Some(name) if !name.is_empty() => 0,
        _ => sce_kernel_error(SCE_KERNEL_ERROR_INVALID_ARGUMENT),
    }
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
    if let Err(error) = memory.protect(
        GuestAddress::new(aligned_address),
        aligned_length,
        protection,
    ) {
        return kernel_memory_error(&error);
    }
    refresh_kernel_memory_readable_ranges(memory);
    if trace_guest_vm() {
        eprintln!(
            "[REMu][VM] protected address={aligned_address:#x} length={aligned_length:#x} protection={protection:#x}"
        );
    }
    0
}

#[cfg(target_os = "macos")]
const fn process_mapping_contains(
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
const fn host_protection_from_guest(raw: i32) -> i32 {
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
    let (mapping_flags, name): (u8, &[u8]) = match allocation.kind {
        kernel_memory::KernelVirtualMappingKind::Reserved => (0x10, b"reserved"),
        kernel_memory::KernelVirtualMappingKind::Flexible => (0x11, b"flexible"),
        kernel_memory::KernelVirtualMappingKind::Direct => (0x12, b"direct"),
    };
    write_guest_u8(output + 0x20, mapping_flags);
    for (index, byte) in name.iter().copied().enumerate() {
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
    refresh_kernel_memory_readable_ranges(memory);
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

fn trace_guest_vfs() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| std::env::var_os("REMU_TRACE_VFS").is_some_and(|value| value == "1"))
}

fn trace_agc_shaders() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| {
        std::env::var_os("REMU_TRACE_AGC_SHADERS").is_some_and(|value| value == "1")
    })
}

fn trace_agc_target_address() -> Option<u64> {
    static ADDRESS: OnceLock<Option<u64>> = OnceLock::new();
    *ADDRESS.get_or_init(|| {
        let value = std::env::var("REMU_TRACE_AGC_TARGET").ok()?;
        let value = value
            .strip_prefix("0x")
            .or_else(|| value.strip_prefix("0X"))
            .unwrap_or(&value);
        u64::from_str_radix(value, 16).ok()
    })
}

fn trace_agc_draws() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED
        .get_or_init(|| std::env::var_os("REMU_TRACE_AGC_DRAWS").is_some_and(|value| value == "1"))
}

fn trace_agc_registers() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| {
        std::env::var_os("REMU_TRACE_AGC_REGISTERS").is_some_and(|value| value == "1")
    })
}

fn trace_guest_time() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| std::env::var_os("REMU_TRACE_TIME").is_some_and(|value| value == "1"))
}

fn trace_ampr_reads() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| std::env::var_os("REMU_TRACE_AMPR").is_some_and(|value| value == "1"))
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

fn guest_mspace_create(arguments: [u64; 6], state: &mut ProbeDispatchState) -> u64 {
    let Some(name) = read_guest_cstring(arguments[0]) else {
        return 0;
    };
    let (base, capacity) = (arguments[1], arguments[2]);
    let Some(end) = base.checked_add(capacity) else {
        return 0;
    };
    if base == 0 || capacity == 0 {
        return 0;
    }
    #[cfg(target_os = "macos")]
    {
        let Some(mapping) = state
            .kernel_memory
            .as_ref()
            .and_then(|memory| memory.query(GuestAddress::new(base), false))
        else {
            return 0;
        };
        let mapping_end = mapping.address.get().saturating_add(mapping.size);
        if base < mapping.address.get() || end > mapping_end {
            return 0;
        }
    }
    let Some(mspace) = GuestMspace::new(name, base, capacity) else {
        return 0;
    };
    if state.guest_mspaces.contains_key(&base) {
        return 0;
    }
    state.guest_mspaces.insert(base, mspace);
    if let Some(mspace) = state.guest_mspaces.get(&base) {
        eprintln!(
            "[REMu][MSPACE] created name={:?} base={:#x} end={:#x}",
            mspace.name, mspace.base, mspace.end
        );
    }
    base
}

fn guest_mspace_malloc(arguments: [u64; 6], state: &mut ProbeDispatchState) -> u64 {
    state
        .guest_mspaces
        .get_mut(&arguments[0])
        .and_then(|mspace| mspace.allocate(arguments[1], GuestMspace::ALLOCATION_ALIGNMENT))
        .unwrap_or(0)
}

fn guest_mspace_calloc(arguments: [u64; 6], state: &mut ProbeDispatchState) -> u64 {
    let Some(size) = arguments[1].checked_mul(arguments[2]) else {
        return 0;
    };
    let address = state
        .guest_mspaces
        .get_mut(&arguments[0])
        .and_then(|mspace| mspace.allocate(size, GuestMspace::ALLOCATION_ALIGNMENT))
        .unwrap_or(0);
    if address != 0 {
        let Ok(size) = usize::try_from(size) else {
            return 0;
        };
        // SAFETY: The mspace allocator returned a live block of at least `size` bytes.
        unsafe { std::ptr::write_bytes(address as *mut u8, 0, size) };
    }
    address
}

fn guest_mspace_free(arguments: [u64; 6], state: &mut ProbeDispatchState) -> u64 {
    if let Some(mspace) = state.guest_mspaces.get_mut(&arguments[0]) {
        mspace.free(arguments[1]);
    }
    0
}

fn guest_mspace_destroy(arguments: [u64; 6], state: &mut ProbeDispatchState) -> u64 {
    if state.guest_mspaces.remove(&arguments[0]).is_some() {
        0
    } else {
        u64::MAX
    }
}

fn guest_mspace_realloc(arguments: [u64; 6], state: &mut ProbeDispatchState) -> u64 {
    state
        .guest_mspaces
        .get_mut(&arguments[0])
        .and_then(|mspace| mspace.realloc(arguments[1], arguments[2]))
        .unwrap_or(0)
}

fn guest_mspace_memalign(arguments: [u64; 6], state: &mut ProbeDispatchState) -> u64 {
    state
        .guest_mspaces
        .get_mut(&arguments[0])
        .and_then(|mspace| mspace.allocate(arguments[2], arguments[1]))
        .unwrap_or(0)
}

fn guest_mspace_posix_memalign(arguments: [u64; 6], state: &mut ProbeDispatchState) -> u64 {
    let (handle, output, alignment, size) =
        (arguments[0], arguments[1], arguments[2], arguments[3]);
    if output == 0
        || !alignment.is_power_of_two()
        || alignment < u64::try_from(std::mem::size_of::<usize>()).unwrap_or(8)
    {
        return u64::from(libc::EINVAL.cast_unsigned());
    }
    let Some(address) = state
        .guest_mspaces
        .get_mut(&handle)
        .and_then(|mspace| mspace.allocate(size, alignment))
    else {
        return u64::from(libc::ENOMEM.cast_unsigned());
    };
    write_guest_u64(output, address);
    0
}

fn guest_mspace_usable_size(arguments: [u64; 6], state: &ProbeDispatchState) -> u64 {
    state
        .guest_mspaces
        .get(&arguments[0])
        .map_or(0, |mspace| mspace.usable_size(arguments[1]))
}

fn guest_mspace_is_heap_empty(arguments: [u64; 6], state: &ProbeDispatchState) -> u64 {
    state
        .guest_mspaces
        .get(&arguments[0])
        .map_or(0, |mspace| u64::from(mspace.allocations.is_empty()))
}

fn json_init_parameter_set_allocator(arguments: [u64; 6], state: &mut ProbeDispatchState) -> u64 {
    let (parameter, allocator, user_data) = (arguments[0], arguments[1], arguments[2]);
    if parameter == 0 || allocator == 0 {
        return sce_kernel_error(SCE_KERNEL_ERROR_INVALID_ARGUMENT);
    }
    state
        .json_runtime
        .allocators
        .insert(parameter, (allocator, user_data));
    write_guest_u64(parameter, allocator);
    write_guest_u64(parameter + 8, user_data);
    0
}

fn json_init_parameter_set_file_buffer_size(
    arguments: [u64; 6],
    state: &mut ProbeDispatchState,
) -> u64 {
    let (parameter, size) = (arguments[0], arguments[1]);
    if parameter == 0 || size == 0 {
        return sce_kernel_error(SCE_KERNEL_ERROR_INVALID_ARGUMENT);
    }
    state.json_runtime.file_buffer_sizes.insert(parameter, size);
    write_guest_u64(parameter + 16, size);
    0
}

fn json_initializer_initialize(arguments: [u64; 6], state: &mut ProbeDispatchState) -> u64 {
    let (initializer, parameter) = (arguments[0], arguments[1]);
    if initializer == 0 || parameter == 0 {
        return sce_kernel_error(SCE_KERNEL_ERROR_INVALID_ARGUMENT);
    }
    state
        .json_runtime
        .initialized_objects
        .insert(initializer, parameter);
    0
}

fn json_initializer_terminate(arguments: [u64; 6], state: &mut ProbeDispatchState) -> u64 {
    if state
        .json_runtime
        .initialized_objects
        .remove(&arguments[0])
        .is_some()
    {
        0
    } else {
        sce_kernel_error(SCE_KERNEL_ERROR_INVALID_ARGUMENT)
    }
}

const AV_PLAYER_ERROR_INVALID_PARAMS: u32 = 0x806a_0001;
const AV_PLAYER_ERROR_OPERATION_FAILED: u32 = 0x806a_0002;

fn av_player_tool(name: &str, environment: &str) -> Option<PathBuf> {
    if let Some(configured) = std::env::var_os(environment).map(PathBuf::from)
        && configured.is_file()
    {
        return Some(configured);
    }
    std::env::var_os("PATH")
        .into_iter()
        .flat_map(|path| std::env::split_paths(&path).collect::<Vec<_>>())
        .map(|directory| directory.join(name))
        .chain(
            ["/opt/homebrew/bin", "/usr/local/bin", "/usr/bin"]
                .into_iter()
                .map(Path::new)
                .map(|directory| directory.join(name)),
        )
        .find(|candidate| candidate.is_file())
}

fn av_player_probe_video(path: &Path) -> Option<(u32, u32, f64, Duration)> {
    let ffprobe = av_player_tool("ffprobe", "REMU_FFPROBE")?;
    let output = Command::new(ffprobe)
        .args([
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=width,height,avg_frame_rate,duration",
            "-of",
            "default=noprint_wrappers=1",
        ])
        .arg(path)
        .output()
        .ok()?;
    if !output.status.success() {
        return None;
    }
    let fields = String::from_utf8(output.stdout).ok()?;
    let mut width = 0;
    let mut height = 0;
    let mut frames_per_second = 30.0;
    let mut duration = Duration::ZERO;
    for line in fields.lines() {
        let Some((name, value)) = line.split_once('=') else {
            continue;
        };
        match name {
            "width" => width = value.parse().ok()?,
            "height" => height = value.parse().ok()?,
            "avg_frame_rate" => {
                let (numerator, denominator) = value.split_once('/').unwrap_or((value, "1"));
                let (Ok(numerator), Ok(denominator)) =
                    (numerator.parse::<f64>(), denominator.parse::<f64>())
                else {
                    continue;
                };
                if numerator.is_finite() && denominator.is_finite() && denominator > 0.0 {
                    frames_per_second = numerator / denominator;
                }
            }
            "duration" => {
                if let Ok(seconds) = value.parse::<f64>()
                    && seconds.is_finite()
                    && seconds >= 0.0
                {
                    duration = Duration::from_secs_f64(seconds);
                }
            }
            _ => {}
        }
    }
    (width != 0 && height != 0 && frames_per_second.is_finite() && frames_per_second > 0.0)
        .then_some((width, height, frames_per_second, duration))
}

fn av_player_resolve_source(game_root: &Path, source: &str) -> Option<PathBuf> {
    let mut normalized = source.trim().trim_end_matches('\0').replace('\\', "/");
    if let Some(file) = normalized.strip_prefix("file://") {
        normalized = file.to_owned();
    } else if let Some(file) = normalized.strip_prefix("file:") {
        normalized = file.to_owned();
    }
    resolve_guest_path(game_root, &normalized).or_else(|| {
        let media = normalized.find("Media/")?;
        resolve_guest_path(game_root, &format!("/app0/{}", &normalized[media..]))
    })
}

fn av_player_set_log_callback(arguments: [u64; 6], state: &mut ProbeDispatchState) -> u64 {
    state.av_player.log_callback = (arguments[0] != 0).then_some((arguments[0], arguments[1]));
    0
}

fn av_player_init_ex(arguments: [u64; 6], state: &mut ProbeDispatchState) -> u64 {
    const MINIMUM_INITIALIZATION_SIZE: usize = 48;
    const MAXIMUM_INITIALIZATION_SIZE: usize = 0x400;
    let (parameter, output) = (arguments[0], arguments[1]);
    if parameter == 0 || output == 0 {
        eprintln!(
            "[REMu][AVPLAYER] InitEx rejected null parameter={parameter:#x} output={output:#x}"
        );
        return sce_kernel_error(AV_PLAYER_ERROR_INVALID_PARAMS);
    }
    let raw_size = read_guest_u64(parameter);
    let Ok(size) = usize::try_from(raw_size) else {
        eprintln!("[REMu][AVPLAYER] InitEx rejected non-host size={raw_size:#x}");
        return sce_kernel_error(AV_PLAYER_ERROR_INVALID_PARAMS);
    };
    if !(MINIMUM_INITIALIZATION_SIZE..=MAXIMUM_INITIALIZATION_SIZE).contains(&size) {
        eprintln!(
            "[REMu][AVPLAYER] InitEx rejected size={size:#x} parameter={parameter:#x} output={output:#x}"
        );
        return sce_kernel_error(AV_PLAYER_ERROR_INVALID_PARAMS);
    }
    let allocator_callbacks = [16_u64, 24, 32, 40].map(|offset| read_guest_u64(parameter + offset));
    if allocator_callbacks.contains(&0) {
        eprintln!(
            "[REMu][AVPLAYER] InitEx rejected allocator callbacks={allocator_callbacks:#x?} size={size:#x}"
        );
        return sce_kernel_error(AV_PLAYER_ERROR_INVALID_PARAMS);
    }
    // SAFETY: the leading bounded size field describes a caller-owned initialization structure.
    let initialization =
        unsafe { std::slice::from_raw_parts(parameter as *const u8, size).to_vec() };
    let handle = state.av_player.next_handle.max(1);
    state.av_player.next_handle = handle.saturating_add(1);
    state.av_player.players.insert(
        handle,
        AvPlayerInstance {
            _initialization: initialization,
            allocator_object: read_guest_u64(parameter + 8),
            allocate_texture_callback: read_guest_u64(parameter + 32),
            event_object: read_guest_u64(parameter + 88),
            event_callback: read_guest_u64(parameter + 96),
            source: None,
            auto_start: read_guest_u8(parameter + 164) != 0,
            width: 0,
            height: 0,
            frames_per_second: 30.0,
            duration: Duration::ZERO,
            paused: false,
            end_of_stream: false,
            playing_since: None,
            elapsed: Duration::ZERO,
            looping: false,
            video_decoder: None,
            video_output: None,
            raw_video_frame: Vec::new(),
            padded_video_frame: Vec::new(),
            video_buffer_base: 0,
            video_buffers: [0; 3],
            video_buffer_stride: 0,
            next_video_buffer: 0,
            next_video_frame: 0,
            audio_decoder: None,
            audio_output: None,
            raw_audio_frame: Vec::new(),
            audio_buffer_base: 0,
            next_audio_buffer: 0,
            next_audio_frame: 0,
        },
    );
    write_guest_u64(output, handle);
    let player = &state.av_player.players[&handle];
    eprintln!(
        "[REMu][AVPLAYER] initialized handle={handle:#x} size={size:#x} texture_alloc={:#x}/{:#x} event={:#x}/{:#x}",
        player.allocator_object,
        player.allocate_texture_callback,
        player.event_object,
        player.event_callback
    );
    0
}

fn av_player_post_init(arguments: [u64; 6], state: &ProbeDispatchState) -> u64 {
    let (handle, parameter) = (arguments[0], arguments[1]);
    if parameter == 0 || !state.av_player.players.contains_key(&handle) {
        return sce_kernel_error(AV_PLAYER_ERROR_INVALID_PARAMS);
    }
    eprintln!("[REMu][AVPLAYER] post-initialized handle={handle:#x}");
    0
}

fn av_player_queue_event(function: u64, object: u64, event: u64, state: &mut ProbeDispatchState) {
    if function == 0 {
        eprintln!("[REMu][AVPLAYER] skipped event={event} without callback");
        return;
    }
    state
        .av_player
        .pending_callbacks
        .push(AvPlayerEventCallback {
            function,
            object,
            event,
        });
}

fn invoke_av_player_event_callbacks(callbacks: Vec<AvPlayerEventCallback>) {
    for callback in callbacks {
        if !process_guest_range_is_readable(callback.function, 1) {
            eprintln!(
                "[REMu][AVPLAYER] rejected unreadable event={} callback={:#x}",
                callback.event, callback.function
            );
            continue;
        }
        type Callback = unsafe extern "C" fn(u64, u64, u64, u64, u64);
        // SAFETY: InitEx supplies a live guest SysV event callback and opaque event object. The
        // process-wide dispatch lock has already been released, so this callback may re-enter any
        // AVPlayer export exactly as the native library permits.
        let function: Callback = unsafe { std::mem::transmute(callback.function) };
        // SAFETY: The callback ABI is `(object, event, source_id, event_data, event_data_size)`.
        unsafe { function(callback.object, callback.event, 0, 0, 0) };
        eprintln!(
            "[REMu][AVPLAYER] delivered event={} callback={:#x}",
            callback.event, callback.function
        );
    }
}

fn prepare_av_player_video_buffers(handle: u64) {
    let Some(dispatch) = PROBE_DISPATCH.get() else {
        return;
    };
    let request = {
        let state = dispatch
            .read()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let Some(player) = state.av_player.players.get(&handle) else {
            return;
        };
        if player.video_buffers[0] != 0
            || player.allocate_texture_callback == 0
            || player.width == 0
            || player.height == 0
        {
            return;
        }
        let Some((_, stride)) = av_player_nv12_layout(player.width, player.height) else {
            return;
        };
        (
            player.allocate_texture_callback,
            player.allocator_object,
            stride,
        )
    };
    if !process_guest_range_is_readable(request.0, 1) {
        eprintln!(
            "[REMu][AVPLAYER] texture allocator callback is unreadable: {:#x}",
            request.0
        );
        return;
    }
    type AllocateTexture = unsafe extern "C" fn(u64, u64, u64, u64, u64) -> u64;
    // SAFETY: InitEx supplies the live guest texture-allocation callback and opaque allocator
    // object. This runs before the process dispatch lock is acquired, so the allocator may call
    // kernel memory exports while it obtains GPU-visible storage.
    let allocate: AllocateTexture = unsafe { std::mem::transmute(request.0) };
    let mut buffers = [0_u64; 3];
    for (index, buffer) in buffers.iter_mut().enumerate() {
        // SAFETY: The callback ABI is `(object, alignment, size, memory_type, reserved)`.
        *buffer = unsafe {
            allocate(
                request.1,
                0x100,
                u64::try_from(request.2).unwrap_or(u64::MAX),
                0,
                0,
            )
        };
        if *buffer == 0 {
            eprintln!("[REMu][AVPLAYER] texture allocator returned null for buffer {index}");
            return;
        }
    }
    let mut state = dispatch
        .write()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let Some(player) = state.av_player.players.get_mut(&handle) else {
        return;
    };
    if player.video_buffers[0] == 0 {
        player.video_buffers = buffers;
        player.video_buffer_stride = request.2;
        eprintln!(
            "[REMu][AVPLAYER] allocated guest video buffers size={} addresses={buffers:#x?}",
            request.2
        );
    }
}

fn av_player_add_source(arguments: [u64; 6], state: &mut ProbeDispatchState) -> u64 {
    let Some(source) = read_guest_cstring(arguments[1]) else {
        return sce_kernel_error(AV_PLAYER_ERROR_INVALID_PARAMS);
    };
    let Some(host_path) = av_player_resolve_source(&state.game_root, &source) else {
        eprintln!("[REMu][AVPLAYER] rejected source outside app0: {source:?}");
        return sce_kernel_error(AV_PLAYER_ERROR_OPERATION_FAILED);
    };
    let Some((width, height, frames_per_second, duration)) = av_player_probe_video(&host_path)
    else {
        eprintln!(
            "[REMu][AVPLAYER] could not probe source guest={source:?} host={}",
            host_path.display()
        );
        return sce_kernel_error(AV_PLAYER_ERROR_OPERATION_FAILED);
    };
    let Some(player) = state.av_player.players.get_mut(&arguments[0]) else {
        return sce_kernel_error(AV_PLAYER_ERROR_INVALID_PARAMS);
    };
    player.reset_playback();
    player.source = Some(host_path.to_string_lossy().into_owned());
    player.width = width;
    player.height = height;
    player.frames_per_second = frames_per_second;
    player.duration = duration;
    player.paused = false;
    if player.auto_start {
        player.playing_since = Some(Instant::now());
    }
    let callback = (
        player.event_callback,
        player.event_object,
        player.auto_start,
    );
    eprintln!(
        "[REMu][AVPLAYER] source handle={:#x} guest={source:?} host={} {}x{} fps={frames_per_second:.3} duration_ms={} auto_start={}",
        arguments[0],
        host_path.display(),
        width,
        height,
        duration.as_millis(),
        player.auto_start,
    );
    av_player_queue_event(callback.0, callback.1, 2, state);
    if callback.2 {
        av_player_queue_event(callback.0, callback.1, 3, state);
    }
    0
}

fn av_player_add_source_ex(arguments: [u64; 6], state: &mut ProbeDispatchState) -> u64 {
    let details = arguments[2];
    if arguments[1] != 0 || details == 0 {
        return sce_kernel_error(AV_PLAYER_ERROR_INVALID_PARAMS);
    }
    let source_address = read_guest_u64(details);
    let length = u64::from(read_guest_u32(details + 8));
    let Ok(length) = usize::try_from(length) else {
        return sce_kernel_error(AV_PLAYER_ERROR_INVALID_PARAMS);
    };
    if source_address == 0 || length == 0 {
        return sce_kernel_error(AV_PLAYER_ERROR_INVALID_PARAMS);
    }
    // SAFETY: the source-details URI pairs this pointer with the exact readable byte length.
    let bytes = unsafe { std::slice::from_raw_parts(source_address as *const u8, length) };
    let Ok(source) = std::str::from_utf8(bytes) else {
        return sce_kernel_error(AV_PLAYER_ERROR_INVALID_PARAMS);
    };
    let terminated = format!("{}\0", source.trim_end_matches('\0'));
    let forwarded = [arguments[0], terminated.as_ptr() as u64, 0, 0, 0, 0];
    av_player_add_source(forwarded, state)
}

fn av_player_start_decoder(player: &mut AvPlayerInstance, audio: bool) -> bool {
    if if audio {
        player.audio_output.is_some()
    } else {
        player.video_output.is_some()
    } {
        return true;
    }
    let (Some(ffmpeg), Some(source)) = (
        av_player_tool("ffmpeg", "REMU_FFMPEG"),
        player.source.as_deref(),
    ) else {
        return false;
    };
    let mut command = Command::new(ffmpeg);
    command
        .args(["-nostdin", "-hide_banner", "-loglevel", "error", "-i"])
        .arg(source);
    if audio {
        command.args([
            "-map", "0:a:0", "-vn", "-ac", "2", "-ar", "48000", "-f", "s16le", "pipe:1",
        ]);
    } else {
        command.args([
            "-map", "0:v:0", "-an", "-pix_fmt", "nv12", "-f", "rawvideo", "pipe:1",
        ]);
    }
    let Ok(mut child) = command.stdout(Stdio::piped()).stderr(Stdio::null()).spawn() else {
        return false;
    };
    let Some(output) = child.stdout.take() else {
        let _ = child.kill();
        let _ = child.wait();
        return false;
    };
    if audio {
        player.raw_audio_frame.resize(1024 * 2 * 2, 0);
        player.audio_output = Some(output);
        player.audio_decoder = Some(child);
    } else {
        let frame_bytes = u64::from(player.width)
            .checked_mul(u64::from(player.height))
            .and_then(|pixels| pixels.checked_mul(3))
            .and_then(|bytes| bytes.checked_div(2))
            .and_then(|bytes| usize::try_from(bytes).ok());
        let Some(frame_bytes) = frame_bytes else {
            let _ = child.kill();
            let _ = child.wait();
            return false;
        };
        player.raw_video_frame.resize(frame_bytes, 0);
        player.video_output = Some(output);
        player.video_decoder = Some(child);
    }
    player.playing_since.get_or_insert_with(Instant::now);
    eprintln!(
        "[REMu][AVPLAYER] started {} decoder source={source:?}",
        if audio { "audio" } else { "video" }
    );
    true
}

fn av_player_read_video_frame(player: &mut AvPlayerInstance) -> bool {
    player
        .video_output
        .as_mut()
        .is_some_and(|output| output.read_exact(&mut player.raw_video_frame).is_ok())
}

fn av_player_finish_stream(player: &mut AvPlayerInstance) {
    if player.looping {
        player.reset_playback();
        player.playing_since = Some(Instant::now());
    } else {
        player.end_of_stream = true;
        if let Some(started) = player.playing_since.take() {
            player.elapsed += started.elapsed();
        }
    }
}

fn av_player_nv12_layout(width: u32, height: u32) -> Option<(usize, usize)> {
    let width = usize::try_from(width).ok()?;
    let height = usize::try_from(height).ok()?;
    let row_pitch = width.checked_add(255)? & !255;
    let chroma_rows = height.checked_add(1)?.checked_div(2)?;
    let frame_bytes = row_pitch.checked_mul(height.checked_add(chroma_rows)?)?;
    Some((row_pitch, frame_bytes))
}

fn av_player_ensure_video_buffers(player: &mut AvPlayerInstance) -> bool {
    if player.video_buffers[0] != 0 {
        return true;
    }
    let Some((_, stride)) = av_player_nv12_layout(player.width, player.height) else {
        return false;
    };
    let Some(total) = stride
        .checked_mul(3)
        .and_then(|bytes| u64::try_from(bytes).ok())
    else {
        return false;
    };
    let allocation = guest_aligned_alloc(0x4000, total);
    if allocation == 0 {
        return false;
    }
    // SAFETY: the dedicated aligned allocation has exactly `total` writable bytes.
    unsafe {
        std::ptr::write_bytes(
            allocation as *mut u8,
            0,
            usize::try_from(total).unwrap_or(0),
        )
    };
    player.video_buffer_base = allocation;
    for (index, buffer) in player.video_buffers.iter_mut().enumerate() {
        *buffer = allocation
            + u64::try_from(index)
                .unwrap_or(0)
                .saturating_mul(u64::try_from(stride).unwrap_or(0));
    }
    player.video_buffer_stride = stride;
    eprintln!(
        "[REMu][AVPLAYER] allocated fallback video buffers size={stride} base={allocation:#x}"
    );
    true
}

fn av_player_write_video_frame(
    player: &mut AvPlayerInstance,
    output: u64,
    timestamp_milliseconds: u64,
    extended: bool,
) -> bool {
    if !av_player_ensure_video_buffers(player) {
        return false;
    }
    let Some((row_pitch, frame_bytes)) = av_player_nv12_layout(player.width, player.height) else {
        return false;
    };
    if player.video_buffer_stride < frame_bytes {
        return false;
    }
    let width = usize::try_from(player.width).unwrap_or(0);
    let height = usize::try_from(player.height).unwrap_or(0);
    let frame = if row_pitch == width {
        &player.raw_video_frame
    } else {
        player.padded_video_frame.resize(frame_bytes, 0);
        player.padded_video_frame.fill(0);
        for row in 0..height {
            let source = row * width;
            let destination = row * row_pitch;
            player.padded_video_frame[destination..destination + width]
                .copy_from_slice(&player.raw_video_frame[source..source + width]);
        }
        let source_chroma = width * height;
        let destination_chroma = row_pitch * height;
        for row in 0..height / 2 {
            let source = source_chroma + row * width;
            let destination = destination_chroma + row * row_pitch;
            player.padded_video_frame[destination..destination + width]
                .copy_from_slice(&player.raw_video_frame[source..source + width]);
        }
        &player.padded_video_frame
    };
    let address = player.video_buffers[player.next_video_buffer];
    player.next_video_buffer = (player.next_video_buffer + 1) % 3;
    // SAFETY: the destination is one complete slice of the dedicated three-frame allocation.
    unsafe { std::ptr::copy_nonoverlapping(frame.as_ptr(), address as *mut u8, frame.len()) };
    guest_memset(output, 0, if extended { 104 } else { 40 });
    write_guest_u64(output, address);
    write_guest_u64(output + 16, timestamp_milliseconds);
    write_guest_u32(
        output + 24,
        if extended {
            player.width
        } else {
            u32::try_from(row_pitch).unwrap_or(player.width)
        },
    );
    write_guest_u32(output + 28, player.height);
    write_guest_u32(output + 32, 1.0_f32.to_bits());
    if extended {
        write_guest_u32(
            output + 60,
            u32::try_from(row_pitch).unwrap_or(player.width),
        );
        write_guest_u8(output + 64, 8);
        write_guest_u8(output + 65, 8);
    }
    true
}

fn av_player_close(handle: u64, state: &mut ProbeDispatchState) -> u64 {
    if state.av_player.players.remove(&handle).is_some() {
        0
    } else {
        sce_kernel_error(AV_PLAYER_ERROR_INVALID_PARAMS)
    }
}

fn av_player_current_time(handle: u64, state: &ProbeDispatchState) -> u64 {
    let Some(player) = state.av_player.players.get(&handle) else {
        return sce_kernel_error(AV_PLAYER_ERROR_INVALID_PARAMS);
    };
    u64::try_from(player.playback_elapsed().as_millis()).unwrap_or(u64::MAX)
}

fn av_player_is_active(handle: u64, state: &ProbeDispatchState) -> u64 {
    state.av_player.players.get(&handle).map_or(0, |player| {
        u64::from(
            player.playing_since.is_some()
                && !player.end_of_stream
                && (player.duration.is_zero() || player.playback_elapsed() < player.duration),
        )
    })
}

fn av_player_pause(handle: u64, state: &mut ProbeDispatchState) -> u64 {
    let Some(player) = state.av_player.players.get_mut(&handle) else {
        return sce_kernel_error(AV_PLAYER_ERROR_INVALID_PARAMS);
    };
    if let Some(started) = player.playing_since.take() {
        player.elapsed += started.elapsed();
    }
    player.paused = true;
    let callback = (player.event_callback, player.event_object);
    av_player_queue_event(callback.0, callback.1, 4, state);
    0
}

fn av_player_start(handle: u64, state: &mut ProbeDispatchState) -> u64 {
    let Some(player) = state.av_player.players.get_mut(&handle) else {
        return sce_kernel_error(AV_PLAYER_ERROR_INVALID_PARAMS);
    };
    if player.source.is_none() {
        return sce_kernel_error(AV_PLAYER_ERROR_OPERATION_FAILED);
    }
    player.paused = false;
    player.end_of_stream = false;
    player.playing_since.get_or_insert_with(Instant::now);
    let callback = (player.event_callback, player.event_object);
    eprintln!("[REMu][AVPLAYER] started handle={handle:#x}");
    av_player_queue_event(callback.0, callback.1, 3, state);
    0
}

fn av_player_jump_to_time(arguments: [u64; 6], state: &mut ProbeDispatchState) -> u64 {
    let Some(player) = state.av_player.players.get_mut(&arguments[0]) else {
        return sce_kernel_error(AV_PLAYER_ERROR_INVALID_PARAMS);
    };
    player.stop_decoders();
    player.elapsed = Duration::from_millis(arguments[1]);
    player.next_video_frame = (player.elapsed.as_secs_f64() * player.frames_per_second) as u64;
    player.next_audio_frame = u64::try_from(player.elapsed.as_millis())
        .unwrap_or(u64::MAX)
        .saturating_mul(48_000)
        / (1024 * 1_000);
    if player.playing_since.is_some() {
        player.playing_since = Some(Instant::now());
    }
    0
}

fn av_player_stop(handle: u64, state: &mut ProbeDispatchState) -> u64 {
    let Some(player) = state.av_player.players.get_mut(&handle) else {
        return sce_kernel_error(AV_PLAYER_ERROR_INVALID_PARAMS);
    };
    player.reset_playback();
    player.paused = false;
    let callback = (player.event_callback, player.event_object);
    av_player_queue_event(callback.0, callback.1, 1, state);
    0
}

fn av_player_set_looping(arguments: [u64; 6], state: &mut ProbeDispatchState) -> u64 {
    let Some(player) = state.av_player.players.get_mut(&arguments[0]) else {
        return sce_kernel_error(AV_PLAYER_ERROR_INVALID_PARAMS);
    };
    player.looping = arguments[1] & 1 != 0;
    0
}

fn av_player_get_video_data(
    arguments: [u64; 6],
    extended: bool,
    state: &mut ProbeDispatchState,
) -> u64 {
    let Some(player) = state.av_player.players.get_mut(&arguments[0]) else {
        return 0;
    };
    if arguments[1] == 0
        || player.playing_since.is_none()
        || player.paused
        || player.end_of_stream
        || !av_player_start_decoder(player, false)
    {
        return 0;
    }
    let expected_frame = (player.playback_elapsed().as_secs_f64()
        * player.frames_per_second.max(1.0))
    .floor() as u64;
    while player.next_video_frame < expected_frame {
        if !av_player_read_video_frame(player) {
            av_player_finish_stream(player);
            return 0;
        }
        player.next_video_frame = player.next_video_frame.saturating_add(1);
    }
    if !av_player_read_video_frame(player) {
        av_player_finish_stream(player);
        return 0;
    }
    let timestamp = ((player.next_video_frame as f64) * 1_000.0 / player.frames_per_second.max(1.0))
        .round() as u64;
    player.next_video_frame = player.next_video_frame.saturating_add(1);
    if av_player_write_video_frame(player, arguments[1], timestamp, extended) {
        if player.next_video_frame <= 4 || player.next_video_frame.is_power_of_two() {
            eprintln!(
                "[REMu][AVPLAYER] video frame={} timestamp_ms={timestamp} extended={extended}",
                player.next_video_frame
            );
        }
        1
    } else {
        0
    }
}

fn av_player_get_audio_data(arguments: [u64; 6], state: &mut ProbeDispatchState) -> u64 {
    const SAMPLES_PER_FRAME: usize = 1024;
    const CHANNELS: usize = 2;
    const SAMPLE_RATE: u32 = 48_000;
    const FRAME_BYTES: usize = SAMPLES_PER_FRAME * CHANNELS * 2;
    let Some(player) = state.av_player.players.get_mut(&arguments[0]) else {
        return 0;
    };
    if arguments[1] == 0
        || player.playing_since.is_none()
        || player.paused
        || player.end_of_stream
        || !av_player_start_decoder(player, true)
        || player
            .audio_output
            .as_mut()
            .is_none_or(|output| output.read_exact(&mut player.raw_audio_frame).is_err())
    {
        return 0;
    }
    if player.audio_buffer_base == 0 {
        let total = u64::try_from(FRAME_BYTES * 8).unwrap_or(0);
        player.audio_buffer_base = guest_aligned_alloc(0x100, total);
        if player.audio_buffer_base == 0 {
            return 0;
        }
    }
    let address = player.audio_buffer_base
        + u64::try_from(player.next_audio_buffer * FRAME_BYTES).unwrap_or(0);
    player.next_audio_buffer = (player.next_audio_buffer + 1) % 8;
    // SAFETY: `address` selects one complete frame in the dedicated eight-frame allocation.
    unsafe {
        std::ptr::copy_nonoverlapping(
            player.raw_audio_frame.as_ptr(),
            address as *mut u8,
            FRAME_BYTES,
        )
    };
    let timestamp = player
        .next_audio_frame
        .saturating_mul(SAMPLES_PER_FRAME as u64)
        * 1_000
        / u64::from(SAMPLE_RATE);
    player.next_audio_frame = player.next_audio_frame.saturating_add(1);
    guest_memset(arguments[1], 0, 40);
    write_guest_u64(arguments[1], address);
    write_guest_u64(arguments[1] + 16, timestamp);
    write_guest_u16(arguments[1] + 24, CHANNELS as u16);
    write_guest_u32(arguments[1] + 28, SAMPLE_RATE);
    write_guest_u32(arguments[1] + 32, FRAME_BYTES as u32);
    1
}

fn av_player_stream_count(handle: u64, state: &ProbeDispatchState) -> u64 {
    if state.av_player.players.contains_key(&handle) {
        2
    } else {
        sce_kernel_error(AV_PLAYER_ERROR_INVALID_PARAMS)
    }
}

fn av_player_validate_handle(handle: u64, state: &ProbeDispatchState) -> u64 {
    if state.av_player.players.contains_key(&handle) {
        0
    } else {
        sce_kernel_error(AV_PLAYER_ERROR_INVALID_PARAMS)
    }
}

fn av_player_get_stream_info(arguments: [u64; 6], state: &ProbeDispatchState) -> u64 {
    let Some(player) = state.av_player.players.get(&arguments[0]) else {
        return sce_kernel_error(AV_PLAYER_ERROR_INVALID_PARAMS);
    };
    if arguments[1] > 1 || arguments[2] == 0 || player.width == 0 || player.height == 0 {
        return sce_kernel_error(AV_PLAYER_ERROR_INVALID_PARAMS);
    }
    guest_memset(arguments[2], 0, 32);
    write_guest_u32(arguments[2], low_u32(arguments[1]));
    if arguments[1] == 0 {
        write_guest_u32(arguments[2] + 8, player.width);
        write_guest_u32(arguments[2] + 12, player.height);
        write_guest_u32(
            arguments[2] + 16,
            ((player.width as f32) / (player.height as f32)).to_bits(),
        );
    } else {
        write_guest_u16(arguments[2] + 8, 2);
        write_guest_u32(arguments[2] + 12, 48_000);
    }
    write_guest_u64(
        arguments[2] + 24,
        u64::try_from(player.duration.as_millis()).unwrap_or(u64::MAX),
    );
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
    if trace_guest_vfs() {
        eprintln!(
            "[REMu][VFS] fopen guest={guest_path:?} host={:?} result={file:p}",
            host_path.to_string_lossy()
        );
    }
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

const SCE_KERNEL_ERROR_OPERATION_NOT_PERMITTED: u32 = 0x8002_0001;
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
const SCE_KERNEL_ERROR_NAME_TOO_LONG: u32 = 0x8002_003f;
const SCE_KERNEL_ERROR_TIMED_OUT: u32 = 0x8002_003c;
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
    let is_directory = file.metadata().is_ok_and(|metadata| metadata.is_dir());
    let mut files = KERNEL_FILES
        .get_or_init(|| Mutex::new(KernelFileState::default()))
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let fd = files.next_fd.max(3);
    files.next_fd = fd.saturating_add(1);
    if is_directory {
        let mut entries: Vec<(String, bool)> = match std::fs::read_dir(&host_path) {
            Ok(listing) => listing
                .filter_map(|entry| {
                    let entry = entry.ok()?;
                    Some((
                        entry.file_name().to_string_lossy().into_owned(),
                        entry.file_type().is_ok_and(|kind| kind.is_dir()),
                    ))
                })
                .collect(),
            Err(error) => {
                eprintln!(
                    "[REMu][VFS] sceKernelOpen could not list guest={guest_path:?} host={:?}: {error}",
                    host_path.display()
                );
                return sce_kernel_io_error(&error);
            }
        };
        // A stable order keeps repeated enumerations reproducible across runs and hosts.
        entries.sort_by(|left, right| left.0.cmp(&right.0));
        let mut next_file_number = files.next_file_number;
        let directory = KernelDirectory::new(&entries, &mut next_file_number);
        files.next_file_number = next_file_number;
        files.directories.insert(fd, directory);
        if trace_guest_vfs() {
            eprintln!(
                "[REMu][VFS] sceKernelOpen directory guest={guest_path:?} entries={} fd={fd}",
                entries.len()
            );
        }
    }
    files.files.insert(fd, file);
    if trace_guest_vfs() {
        eprintln!(
            "[REMu][VFS] sceKernelOpen guest={guest_path:?} host={:?} flags={flags:#x} fd={fd}",
            host_path.display()
        );
    }
    signed_i32_return(fd)
}

/// Reads packed directory entries into the guest buffer.
///
/// Returns the number of bytes written and zero once the enumeration is exhausted. `basep`, when
/// supplied, receives the enumeration position before the read so the guest can resume there.
fn guest_kernel_getdents(arguments: [u64; 6], report_position: bool) -> u64 {
    let (fd, buffer, capacity) = (low_i32(arguments[0]), arguments[1], arguments[2]);
    if buffer == 0 {
        return sce_kernel_error(SCE_KERNEL_ERROR_MEMORY_FAULT);
    }
    if capacity < DIRENT_BLOCK_SIZE as u64 {
        return sce_kernel_error(SCE_KERNEL_ERROR_INVALID_ARGUMENT);
    }
    let mut files = KERNEL_FILES
        .get_or_init(|| Mutex::new(KernelFileState::default()))
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    if !files.files.contains_key(&fd) {
        return sce_kernel_error(SCE_KERNEL_ERROR_BAD_FILE_DESCRIPTOR);
    }
    let Some(directory) = files.directories.get_mut(&fd) else {
        return sce_kernel_error(SCE_KERNEL_ERROR_INVALID_ARGUMENT);
    };
    if report_position && arguments[3] != 0 {
        write_guest_u64(arguments[3], directory.offset as u64);
    }
    let remaining = directory.records.len().saturating_sub(directory.offset);
    if remaining == 0 {
        return 0;
    }
    // Only whole blocks are transferred, so a record is never split across two reads.
    let transfer = remaining.min((capacity as usize / DIRENT_BLOCK_SIZE) * DIRENT_BLOCK_SIZE);
    if transfer == 0 {
        return sce_kernel_error(SCE_KERNEL_ERROR_INVALID_ARGUMENT);
    }
    let end = directory.offset + transfer;
    write_guest_bytes(buffer, &directory.records[directory.offset..end]);
    directory.offset = end;
    transfer as u64
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
    if trace_guest_vfs() {
        eprintln!(
            "[REMu][VFS] sceKernelStat guest={guest_path:?} host={:?} size={}",
            host_path.display(),
            metadata.len()
        );
    }
    0
}

fn guest_kernel_read(arguments: [u64; 6]) -> u64 {
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
    let mut files = KERNEL_FILES
        .get_or_init(|| Mutex::new(KernelFileState::default()))
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let Some(file) = files.files.get_mut(&fd) else {
        return sce_kernel_error(SCE_KERNEL_ERROR_BAD_FILE_DESCRIPTOR);
    };
    // SAFETY: sceKernelRead supplies a writable range of exactly `requested` bytes in live guest
    // memory, and the global import dispatcher serializes access to the associated File.
    let destination = unsafe { std::slice::from_raw_parts_mut(destination as *mut u8, requested) };
    let read = match file.read(destination) {
        Ok(read) => read,
        Err(error) => return sce_kernel_io_error(&error),
    };
    if trace_guest_vfs() {
        eprintln!("[REMu][VFS] sceKernelRead fd={fd} requested={requested} read={read}");
    }
    u64::try_from(read).expect("read byte count fits u64")
}

fn guest_kernel_write(arguments: [u64; 6]) -> u64 {
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
    let mut files = KERNEL_FILES
        .get_or_init(|| Mutex::new(KernelFileState::default()))
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let result = match fd {
        1 => std::io::stdout().write(payload),
        2 => std::io::stderr().write(payload),
        _ => files
            .files
            .get_mut(&fd)
            .ok_or_else(|| std::io::Error::from(std::io::ErrorKind::NotFound))
            .and_then(|file| file.write(payload)),
    };
    match result {
        Ok(written) => u64::try_from(written).expect("written byte count fits u64"),
        Err(error) if fd != 1 && fd != 2 && !files.files.contains_key(&fd) => {
            sce_kernel_error(SCE_KERNEL_ERROR_BAD_FILE_DESCRIPTOR)
        }
        Err(error) => sce_kernel_io_error(&error),
    }
}

fn guest_kernel_close(arguments: [u64; 6]) -> u64 {
    let fd = low_i32(arguments[0]);
    let mut files = KERNEL_FILES
        .get_or_init(|| Mutex::new(KernelFileState::default()))
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    files.directories.remove(&fd);
    if files.files.remove(&fd).is_none() {
        return sce_kernel_error(SCE_KERNEL_ERROR_BAD_FILE_DESCRIPTOR);
    }
    drop(files);
    if trace_guest_vfs() {
        eprintln!("[REMu][VFS] sceKernelClose fd={fd}");
    }
    0
}

fn guest_kernel_lseek(arguments: [u64; 6]) -> u64 {
    let fd = low_i32(arguments[0]);
    let offset = i64::from_ne_bytes(arguments[1].to_ne_bytes());
    let seek_from = match low_i32(arguments[2]) {
        0 if offset >= 0 => std::io::SeekFrom::Start(offset.cast_unsigned()),
        1 => std::io::SeekFrom::Current(offset),
        2 => std::io::SeekFrom::End(offset),
        _ => return sce_kernel_error(SCE_KERNEL_ERROR_INVALID_ARGUMENT),
    };
    let mut files = KERNEL_FILES
        .get_or_init(|| Mutex::new(KernelFileState::default()))
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let Some(file) = files.files.get_mut(&fd) else {
        return sce_kernel_error(SCE_KERNEL_ERROR_BAD_FILE_DESCRIPTOR);
    };
    match file.seek(seek_from) {
        Ok(position) => position,
        Err(error) => sce_kernel_io_error(&error),
    }
}

fn guest_kernel_fstat(arguments: [u64; 6]) -> u64 {
    if arguments[1] == 0 {
        return sce_kernel_error(SCE_KERNEL_ERROR_MEMORY_FAULT);
    }
    let fd = low_i32(arguments[0]);
    let files = KERNEL_FILES
        .get_or_init(|| Mutex::new(KernelFileState::default()))
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let Some(file) = files.files.get(&fd) else {
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

fn guest_kernel_ftruncate(arguments: [u64; 6]) -> u64 {
    let fd = low_i32(arguments[0]);
    let length = i64::from_ne_bytes(arguments[1].to_ne_bytes());
    if length < 0 {
        return sce_kernel_error(SCE_KERNEL_ERROR_INVALID_ARGUMENT);
    }
    let mut files = KERNEL_FILES
        .get_or_init(|| Mutex::new(KernelFileState::default()))
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let Some(file) = files.files.get_mut(&fd) else {
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

fn guest_kernel_gettimeofday(output: u64) -> u64 {
    if output == 0 {
        return sce_kernel_error(SCE_KERNEL_ERROR_INVALID_ARGUMENT);
    }
    let elapsed = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default();
    write_guest_u64(output, elapsed.as_secs());
    write_guest_u64(output + 8, u64::from(elapsed.subsec_micros()));
    0
}

/// Returns the host's UTC offset and daylight-saving component for a Unix timestamp. Console
/// calendar conversions describe the machine's configured civil time zone, so they must not be
/// reduced to UTC-only stubs even though emulated monotonic clocks remain host-independent.
fn guest_local_timezone(unix_seconds: i64) -> (i64, i32) {
    let host_time: libc::time_t = unix_seconds;
    // SAFETY: `local` is writable and `host_time` is a valid scalar time_t value.
    let mut local: libc::tm = unsafe { std::mem::zeroed() };
    let result = unsafe { libc::localtime_r(&raw const host_time, &raw mut local) };
    if result.is_null() {
        return (0, 0);
    }
    #[cfg(any(target_os = "macos", target_os = "ios", target_os = "freebsd"))]
    let offset = local.tm_gmtoff;
    #[cfg(not(any(target_os = "macos", target_os = "ios", target_os = "freebsd")))]
    let offset = 0;
    let daylight_seconds = if local.tm_isdst > 0 { 3_600 } else { 0 };
    (offset, daylight_seconds)
}

fn guest_kernel_convert_localtime_to_utc(arguments: [u64; 6]) -> u64 {
    let local_seconds = i64::from_ne_bytes(arguments[0].to_ne_bytes());
    let utc_output = arguments[2];
    let timezone_output = arguments[3];
    let dst_output = arguments[4];
    if timezone_output == 0 {
        return sce_kernel_error(SCE_KERNEL_ERROR_INVALID_ARGUMENT);
    }
    let (offset_seconds, daylight_seconds) = guest_local_timezone(local_seconds);
    let utc_seconds = local_seconds.saturating_sub(offset_seconds);
    let minutes_west = i32::try_from(-offset_seconds / 60).unwrap_or(0);
    write_guest_u32(
        timezone_output,
        u32::from_ne_bytes(minutes_west.to_ne_bytes()),
    );
    write_guest_u32(
        timezone_output + 4,
        u32::from_ne_bytes((daylight_seconds / 60).to_ne_bytes()),
    );
    if utc_output != 0 {
        write_guest_u64(utc_output, u64::from_ne_bytes(utc_seconds.to_ne_bytes()));
    }
    if dst_output != 0 {
        write_guest_u32(
            dst_output,
            u32::from_ne_bytes(daylight_seconds.to_ne_bytes()),
        );
    }
    0
}

fn guest_kernel_convert_utc_to_localtime(arguments: [u64; 6]) -> u64 {
    let utc_seconds = i64::from_ne_bytes(arguments[0].to_ne_bytes());
    let local_output = arguments[1];
    let timesec_output = arguments[2];
    let dst_output = arguments[3];
    if local_output == 0 {
        return sce_kernel_error(SCE_KERNEL_ERROR_INVALID_ARGUMENT);
    }
    let (offset_seconds, daylight_seconds) = guest_local_timezone(utc_seconds);
    let local_seconds = utc_seconds.saturating_add(offset_seconds);
    write_guest_u64(
        local_output,
        u64::from_ne_bytes(local_seconds.to_ne_bytes()),
    );
    if timesec_output != 0 {
        write_guest_u64(
            timesec_output,
            u64::from_ne_bytes(utc_seconds.to_ne_bytes()),
        );
        write_guest_u32(
            timesec_output + 8,
            u32::try_from(offset_seconds).unwrap_or(0),
        );
        write_guest_u32(
            timesec_output + 12,
            u32::try_from(daylight_seconds).unwrap_or(0),
        );
    }
    if dst_output != 0 {
        write_guest_u64(dst_output, u64::try_from(daylight_seconds).unwrap_or(0));
    }
    0
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
    invoke_guest_module_initializers(&image.module_initializers, 0, 0);
    if timeout_seconds != 0 {
        // SAFETY: Alarm only bounds this diagnostic process; its default action terminates it.
        unsafe {
            libc::alarm(timeout_seconds);
        }
    }
    let return_value = invoke_guest_entry_with_exit(
        image.memory.entry_address().get(),
        (&raw const parameters).addr() as u64,
        guest_exit as *const () as usize as u64,
    ) as i32;
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
    use std::sync::atomic::{AtomicU32, AtomicUsize, Ordering};
    use std::sync::{Arc, Mutex};

    use super::{
        AGC_IT_DISPATCH_DIRECT, AGC_IT_DISPATCH_INDIRECT, AGC_IT_INDEX_BASE,
        AGC_IT_INDEX_BUFFER_SIZE, AGC_IT_NOP, AGC_IT_NUM_INSTANCES, AGC_IT_SET_BASE,
        AGC_IT_SET_CONTEXT_REG_INDIRECT, AGC_IT_SET_SH_REG_INDIRECT, AGC_IT_SET_UCONFIG_REG_INDEX,
        AGC_IT_SET_UCONFIG_REG_INDIRECT, AGC_R_ACB_RESET, AGC_R_DRAW_INDEX_AUTO,
        AgcColorTargetState, AgcComputeDispatchError, AgcGraphicsState, AgcShaderRegistry,
        BATCH_MAP_OPERATION_MAP_DIRECT, GuestExitCallback, KERNEL_FILES, KernelFileState,
        ProbeDispatchState, PthreadAttrState, SAVE_DATA_ERROR_EXISTS, SAVE_DATA_ERROR_NOT_FOUND,
        SAVE_DATA_PARAM_SIZE, SCE_KERNEL_ERROR_INVALID_ARGUMENT, SCE_KERNEL_ERROR_NOT_FOUND,
        SCE_KERNEL_ERROR_NOT_IMPLEMENTED, VideoOutBufferAttributes, VideoOutState,
        agc_apply_set_base, agc_apply_submission_registers, agc_color_target_state,
        agc_dcb_draw_index_auto, agc_dcb_set_index_buffer, agc_dcb_set_num_instances,
        agc_decode_compute_dispatch, agc_driver_submit_acb, agc_pm4, ajm_finalize, ajm_initialize,
        ajm_instance_create, ajm_instance_destroy, ajm_module_register, ajm_module_unregister,
        allocate_direct_memory, allocate_pthread_thread_object,
        ampr_command_buffer_write_kernel_event_queue, ampr_submit_command_buffer,
        civil_from_unix_seconds, clone_guest_tls, create_pthread_attr, decode_agc_memory_acquire,
        destroy_pthread_attr, dispatch_guest_numeric_parser, dispatch_probe_import,
        guest_clock_gettime, guest_free, guest_global_locale,
        guest_kernel_convert_localtime_to_utc, guest_kernel_convert_utc_to_localtime,
        guest_kernel_fstat, guest_kernel_ftruncate, guest_kernel_gettimeofday, guest_kernel_lseek,
        guest_kernel_read, guest_posix_file_result, guest_pthread_once, guest_qsort,
        guest_tolower_table, guest_toupper_table, guest_wmemchr, guest_wmemcmp, guest_wmemcpy,
        guest_wmemmove, kernel_add_ampr_event, kernel_batch_map, kernel_release_direct_memory,
        kernel_stat_payload, kernel_virtual_query, low_f32, low_u32, pthread_attr_get_affinity,
        pthread_attr_get_stack, pthread_attr_set_affinity, pthread_attr_set_detach_state,
        pthread_attr_set_schedule_param, pthread_attr_set_stack_size, read_guest_u16,
        read_guest_u32, read_guest_u64, resolve_guest_path, rtc_get_current_clock_local_time,
        rtc_get_current_tick, rtc_get_tick, rtc_set_tick, sanitize_save_segment, save_data_commit,
        save_data_create_transaction_resource, save_data_dir_name_search, save_data_mount3,
        save_data_pattern_matches, save_data_prepare, save_data_save_icon, save_data_set_param,
        save_data_umount2, sce_kernel_error, set_current_guest_thread_pointer,
        system_service_param_get_int, take_exit_callbacks, tls_dtv_entry_offset,
        unix_days_from_civil, video_out_open, write_guest_u32, write_guest_u64, write_import_stub,
    };
    use super::{AmprCommandBufferState, DIRENT_BLOCK_SIZE, DIRENT_HEADER_SIZE, KernelDirectory};

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
    static GUEST_CONDITION_TEST: Mutex<()> = Mutex::new(());
    static GUEST_EXCEPTION_TEST: Mutex<()> = Mutex::new(());
    static DELIVERED_EXCEPTION: AtomicU32 = AtomicU32::new(0);
    static ONCE_CALLBACKS: AtomicUsize = AtomicUsize::new(0);

    unsafe extern "C" fn once_test_callback() {
        ONCE_CALLBACKS.fetch_add(1, Ordering::Relaxed);
    }

    #[cfg(target_arch = "x86_64")]
    unsafe extern "C" fn pthread_exit_boundary_test_entry(argument0: u64, argument1: u64) -> u64 {
        super::guest_pthread_exit(argument0 + argument1);
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn pthread_exit_returns_to_the_host_thread_boundary() {
        assert_eq!(
            super::invoke_guest_entry_with_exit(
                pthread_exit_boundary_test_entry as *const () as usize as u64,
                20,
                22,
            ),
            42
        );
    }

    static DELIVERED_EXCEPTION_RSP: AtomicUsize = AtomicUsize::new(0);

    unsafe extern "C" fn capture_guest_exception(exception: i32, context: *mut u8) {
        DELIVERED_EXCEPTION.store(exception.cast_unsigned(), Ordering::SeqCst);
        // SAFETY: the delivery test passes a complete 0x500-byte synthesized Orbis ucontext.
        let rsp = unsafe { std::ptr::read_unaligned(context.add(0xf8).cast::<u64>()) };
        DELIVERED_EXCEPTION_RSP.store(usize::try_from(rsp).unwrap(), Ordering::SeqCst);
    }

    #[test]
    fn delivers_completed_agc_submissions_through_the_registered_graphics_equeue() {
        let _serial = GPU_EVENT_SINK_TEST
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let completed = remu_gpu::gpu_completed_submission_sequence();
        let first = completed.saturating_add(1);
        let second = first.saturating_add(1);
        let mut state = ProbeDispatchState::default();
        let mut equeue = 0_u64;
        assert_eq!(
            super::dispatch_stateless_import(
                "D0OdFMjp46I",
                [(&raw mut equeue).addr() as u64, 0, 0, 0, 0, 0],
                [0; 6],
            ),
            None,
            "the stateful equeue creator must not be shadowed by a placeholder"
        );
        assert_eq!(
            super::kernel_create_equeue((&raw mut equeue).addr() as u64, &mut state),
            0
        );
        assert_eq!(equeue, 1);
        assert_eq!(
            super::agc_driver_add_eq_event([equeue, 0x1234, 0xfeed_face, 0, 0, 0], &mut state),
            0
        );
        state.pending_agc_completions.extend([first, second]);
        remu_gpu::record_gpu_submission_completed(second);

        let mut event = [0_u8; 0x40];
        let mut count = 0_u32;
        assert_eq!(
            super::kernel_poll_equeue(
                [
                    equeue,
                    event.as_mut_ptr().addr() as u64,
                    2,
                    (&raw mut count).addr() as u64,
                    0,
                    0,
                ],
                &mut state,
            ),
            Ok(1)
        );
        assert_eq!(count, 1);
        assert_eq!(read_guest_u64(event.as_ptr().addr() as u64), 0x1234);
        assert_eq!(
            read_guest_u16(event.as_ptr().addr() as u64 + 8),
            super::KERNEL_EVENT_FILTER_GRAPHICS.cast_unsigned()
        );
        assert_eq!(read_guest_u32(event.as_ptr().addr() as u64 + 0x0c), 2);
        assert_eq!(read_guest_u64(event.as_ptr().addr() as u64 + 0x10), 0x1234);
        assert_eq!(
            read_guest_u64(event.as_ptr().addr() as u64 + 0x18),
            0xfeed_face
        );
        assert!(state.pending_agc_completions.is_empty());
    }

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
            index_snapshot: None,
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

    fn targetless_scanout_test_video(address: u64) -> VideoOutState {
        VideoOutState {
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
        }
    }

    fn targetless_scanout_test_shader_state(
        pixel_address: u64,
        target: remu_gpu::GpuColorTarget,
    ) -> (AgcGraphicsState, AgcShaderRegistry) {
        let mut graphics = AgcGraphicsState::default();
        graphics.known_color_targets.insert(target.address, target);
        let mut template = targetless_scanout_test_draw();
        template.pixel_shader_address = pixel_address;
        let mut pixel_snapshot = template.pixel_shader.take().expect("test pixel shader");
        pixel_snapshot.program = Arc::new(remu_shader::Program::new(
            pixel_address,
            pixel_snapshot.program.instructions.clone(),
        ));
        graphics.context_registers.extend(
            template
                .context_registers
                .iter()
                .map(|register| (register.offset, register.value)),
        );
        graphics.shader_registers.extend([
            (super::AGC_SPI_SHADER_PGM_LO_PS, low_u32(pixel_address >> 8)),
            (
                super::AGC_SPI_SHADER_PGM_HI_PS,
                low_u32(pixel_address >> 40),
            ),
            (0x00b, 0),
        ]);
        let mut shaders = AgcShaderRegistry::default();
        shaders.snapshots.insert(pixel_address, pixel_snapshot);
        (graphics, shaders)
    }

    fn targetless_barrier_acquire_flip_commands() -> (Vec<u32>, remu_gpu::GpuMemoryAcquire) {
        let acquire = [
            agc_pm4(8, AGC_IT_NOP, super::AGC_R_ACQUIRE_MEM),
            0x8000_1234,
            0x3456_789a,
            0x0000_00ab,
            0x7654_3210,
            0x0000_cdef,
            0xabcd_2468,
            0xfff5_4321,
        ];
        let expected_acquire = decode_agc_memory_acquire(acquire.as_ptr().addr() as u64);
        let mut commands = vec![
            agc_pm4(3, super::AGC_IT_DRAW_INDEX_AUTO, 0),
            3,
            0,
            agc_pm4(2, super::AGC_IT_EVENT_WRITE, 0),
            0x410,
        ];
        commands.extend(acquire);
        commands.extend([
            agc_pm4(6, AGC_IT_NOP, super::AGC_R_FLIP),
            1,
            0,
            0,
            0x1234,
            0,
        ]);
        (commands, expected_acquire)
    }

    #[test]
    fn gpu_progress_telemetry_keeps_a_prefix_and_logarithmic_checkpoints() {
        for sequence in 0..=8 {
            assert!(super::trace_logarithmic_sequence(sequence, 8));
        }
        assert!(!super::trace_logarithmic_sequence(9, 8));
        assert!(super::trace_logarithmic_sequence(16, 8));
        assert!(!super::trace_logarithmic_sequence(17, 8));
        assert!(super::trace_logarithmic_sequence(1_u64 << 63, 8));
    }

    #[test]
    fn condition_signal_selects_one_waiter_and_broadcast_selects_the_rest() {
        let mut condition = super::GuestConditionState::default();
        let mutex = 0x1234;
        let first = 1;
        let second = 2;
        let third = 3;
        assert!(condition.register_waiter(first, mutex));
        assert!(condition.register_waiter(second, mutex));
        assert!(condition.register_waiter(third, mutex));
        assert!(!condition.register_waiter(4, mutex + 1));

        assert_eq!(condition.release_one(), Some(first));
        assert_eq!(
            condition.waiters.iter().copied().collect::<Vec<_>>(),
            [second, third]
        );
        assert_eq!(condition.bound_mutex, Some(mutex));

        assert_eq!(
            condition.release_all().into_iter().collect::<Vec<_>>(),
            [second, third]
        );
        assert!(condition.waiters.is_empty());
        assert_eq!(condition.bound_mutex, None);
        assert!(condition.is_idle());

        assert_eq!(condition.release_one(), None, "signals are not retained");
        let future = 5;
        assert!(condition.register_waiter(future, mutex + 1));
        condition.cancel_wait(future);
        assert!(condition.is_idle());
        assert_eq!(condition.bound_mutex, None);
    }

    #[test]
    fn rwlock_tracks_recursive_read_ownership_and_excludes_writers() {
        let _guard = GUEST_CONDITION_TEST
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let address = 0x7fff_0000_0080;
        assert_eq!(super::guest_rwlock_init(address), 0);
        assert_eq!(super::guest_rwlock_lock(address, false, false), 0);
        assert_eq!(super::guest_rwlock_lock(address, false, false), 0);
        assert_eq!(
            super::guest_rwlock_lock(address, true, true),
            u64::from(libc::EBUSY.cast_unsigned())
        );
        assert_eq!(
            super::guest_rwlock_destroy(address),
            u64::from(libc::EBUSY.cast_unsigned())
        );
        assert_eq!(super::guest_rwlock_unlock(address), 0);
        assert_eq!(super::guest_rwlock_unlock(address), 0);
        assert_eq!(super::guest_rwlock_lock(address, true, false), 0);
        assert_eq!(
            super::guest_rwlock_lock(address, false, true),
            u64::from(libc::EBUSY.cast_unsigned())
        );
        assert_eq!(super::guest_rwlock_unlock(address), 0);
        assert_eq!(super::guest_rwlock_destroy(address), 0);
    }

    #[test]
    fn module_unwind_query_writes_the_gen5_metadata_layout() {
        let mut state = ProbeDispatchState::default();
        state.unwind_modules.push(super::GuestUnwindModule {
            name: "eboot.bin".to_owned(),
            base: 0x1000,
            end: 0x9000,
            eh_frame_header: 0x7000,
            eh_frame: 0x5000,
            eh_frame_size: 0x2000,
        });
        let mut output = vec![0xcc_u8; 0x130];
        let output_address = output.as_mut_ptr().addr() as u64;
        super::write_guest_u64(output_address, 0x130);
        assert_eq!(
            super::kernel_module_info_for_unwind([0x1234, 1, output_address, 0, 0, 0], &state),
            0
        );
        assert_eq!(super::read_guest_u64(output_address), 0x130);
        assert_eq!(&output[8..17], b"eboot.bin");
        assert_eq!(super::read_guest_u64(output_address + 0x108), 0x7000);
        assert_eq!(super::read_guest_u64(output_address + 0x110), 0x5000);
        assert_eq!(super::read_guest_u64(output_address + 0x118), 0x2000);
        assert_eq!(super::read_guest_u64(output_address + 0x120), 0x1000);
        assert_eq!(super::read_guest_u64(output_address + 0x128), 0x8000);
    }

    #[test]
    fn timed_condition_wait_reacquires_the_guest_mutex_before_returning() {
        let _guard = GUEST_CONDITION_TEST
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let condition = 0x7fff_0000_0100;
        let mutex = 0x7fff_0000_0200;
        assert_eq!(super::guest_condition_init(condition), 0);
        assert_eq!(
            super::guest_mutex_init(
                mutex,
                super::PthreadMutexAttrState {
                    mutex_type: 2,
                    ..super::PthreadMutexAttrState::default()
                }
            ),
            0
        );
        assert_eq!(super::guest_mutex_lock(mutex, false), 0);
        assert_eq!(super::guest_mutex_lock(mutex, false), 0);

        assert_eq!(
            super::guest_condition_wait(condition, mutex, Some(0)),
            sce_kernel_error(super::SCE_KERNEL_ERROR_TIMED_OUT)
        );
        {
            let entry = super::guest_mutex_entry(mutex, false).expect("test mutex");
            let guest_mutex = entry
                .state
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            assert_eq!(guest_mutex.owner, Some(std::thread::current().id()));
            assert_eq!(guest_mutex.recursion, 2);
            let state = super::guest_sync()
                .0
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            assert!(
                state
                    .conditions
                    .get(&condition)
                    .is_some_and(super::GuestConditionState::is_idle)
            );
        }

        assert_eq!(super::guest_mutex_unlock(mutex), 0);
        assert_eq!(super::guest_mutex_unlock(mutex), 0);
        assert_eq!(super::guest_mutex_destroy(mutex), 0);
        assert_eq!(super::guest_condition_destroy(condition), 0);
    }

    #[test]
    fn condition_wait_delivers_interruption_before_reacquiring_guest_mutex() {
        let _guard = GUEST_CONDITION_TEST
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        const THREAD: u64 = 0x7fff_3000;
        let condition = 0x7fff_0000_0500;
        let mutex = 0x7fff_0000_0600;
        assert_eq!(super::guest_condition_init(condition), 0);
        assert_eq!(
            super::guest_mutex_init(mutex, super::PthreadMutexAttrState::default()),
            0
        );

        let (handler_entered_tx, handler_entered_rx) = std::sync::mpsc::channel();
        let (resume_handler_tx, resume_handler_rx) = std::sync::mpsc::channel();
        let (finished_tx, finished_rx) = std::sync::mpsc::channel();
        let waiter = std::thread::spawn(move || {
            super::set_current_guest_thread_handle(THREAD);
            assert_eq!(super::guest_mutex_lock(mutex, false), 0);
            assert_eq!(
                super::guest_condition_wait_with_interrupt(condition, mutex, None, || {
                    handler_entered_tx
                        .send(())
                        .expect("test receiver remains live");
                    resume_handler_rx
                        .recv()
                        .expect("test suspension controller remains live");
                }),
                0
            );
            assert_eq!(super::guest_mutex_unlock(mutex), 0);
            super::set_current_guest_thread_handle(0);
            finished_tx.send(()).expect("test receiver remains live");
        });
        wait_for_guest_condition_waiters(condition, 1);

        // Model stop-the-world with another guest thread holding this condition's mutex. The
        // interrupted waiter must enter its suspension handler even though mutex reacquisition
        // cannot yet succeed.
        assert_eq!(super::guest_mutex_lock(mutex, false), 0);
        super::guest_exception_delivery()
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .pending
            .insert(
                THREAD,
                super::PendingGuestException {
                    handler: 1,
                    exception: 30,
                },
            );
        super::guest_sync().1.notify_all();
        handler_entered_rx
            .recv_timeout(std::time::Duration::from_secs(2))
            .expect("interruption handler ran before mutex reacquisition");

        resume_handler_tx
            .send(())
            .expect("waiter remains in its handler");
        assert!(
            finished_rx
                .recv_timeout(std::time::Duration::from_millis(50))
                .is_err(),
            "condition wait returned without reacquiring its mutex"
        );
        assert_eq!(super::guest_mutex_unlock(mutex), 0);
        finished_rx
            .recv_timeout(std::time::Duration::from_secs(2))
            .expect("resumed waiter reacquired the mutex");
        waiter.join().expect("guest waiter thread");

        super::guest_exception_delivery()
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .pending
            .remove(&THREAD);
        assert_eq!(super::guest_mutex_destroy(mutex), 0);
        assert_eq!(super::guest_condition_destroy(condition), 0);
    }

    #[test]
    fn condition_wait_errors_use_the_selected_pthread_abi() {
        let _guard = GUEST_CONDITION_TEST
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let condition = 0x7fff_0000_0900;
        let mutex = 0x7fff_0000_0a00;
        assert_eq!(super::guest_condition_init(condition), 0);
        assert_eq!(
            super::guest_mutex_init(mutex, super::PthreadMutexAttrState::default()),
            0
        );
        assert_eq!(
            super::dispatch_guest_sync_import("WKAXJ4XBPQ4", [condition, mutex, 0, 0, 0, 0], None),
            sce_kernel_error(super::SCE_KERNEL_ERROR_OPERATION_NOT_PERMITTED)
        );
        assert_eq!(
            super::dispatch_guest_sync_import("Op8TBGY5KHg", [condition, mutex, 0, 0, 0, 0], None),
            u64::from(libc::EPERM.cast_unsigned())
        );
        assert_eq!(super::guest_mutex_destroy(mutex), 0);
        assert_eq!(super::guest_condition_destroy(condition), 0);
    }

    #[test]
    fn condition_signal_wakes_exactly_one_blocked_guest_thread() {
        let _guard = GUEST_CONDITION_TEST
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let condition = 0x7fff_0000_1100;
        let mutex = 0x7fff_0000_1200;
        assert_eq!(super::guest_condition_init(condition), 0);
        assert_eq!(
            super::guest_mutex_init(mutex, super::PthreadMutexAttrState::default()),
            0
        );
        let (finished, receiver) = std::sync::mpsc::channel();
        let waiters = (0..2)
            .map(|waiter| {
                let finished = finished.clone();
                std::thread::spawn(move || {
                    assert_eq!(super::guest_mutex_lock(mutex, false), 0);
                    assert_eq!(super::guest_condition_wait(condition, mutex, None), 0);
                    assert_eq!(super::guest_mutex_unlock(mutex), 0);
                    finished.send(waiter).expect("test receiver remains live");
                })
            })
            .collect::<Vec<_>>();
        drop(finished);
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(2);
        loop {
            let waiting = super::guest_sync()
                .0
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner)
                .conditions
                .get(&condition)
                .map_or(0, |condition| condition.waiters.len());
            if waiting == 2 {
                break;
            }
            assert!(
                std::time::Instant::now() < deadline,
                "guest waiters did not block"
            );
            std::thread::yield_now();
        }

        assert_eq!(
            super::dispatch_guest_sync_import("kDh-NfxgMtE", [condition, 0, 0, 0, 0, 0], None),
            0
        );
        let first = receiver
            .recv_timeout(std::time::Duration::from_secs(2))
            .expect("one signaled waiter");
        let early_second = receiver
            .recv_timeout(std::time::Duration::from_millis(50))
            .ok();
        assert_eq!(
            super::dispatch_guest_sync_import("JGgj7Uvrl+A", [condition, 0, 0, 0, 0, 0], None),
            0
        );
        let second = early_second.unwrap_or_else(|| {
            receiver
                .recv_timeout(std::time::Duration::from_secs(2))
                .expect("broadcast waiter")
        });
        assert_ne!(first, second);
        assert!(
            early_second.is_none(),
            "signal acquired broadcast semantics"
        );
        for waiter in waiters {
            waiter.join().expect("guest waiter thread");
        }
        assert_eq!(super::guest_mutex_destroy(mutex), 0);
        assert_eq!(super::guest_condition_destroy(condition), 0);
    }

    #[test]
    fn condition_signal_allows_exactly_one_predicate_guarded_task_consumption() {
        let _guard = GUEST_CONDITION_TEST
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let condition = 0x7fff_0000_1900;
        let mutex = 0x7fff_0000_1a00;
        assert_eq!(super::guest_condition_init(condition), 0);
        assert_eq!(
            super::guest_mutex_init(mutex, super::PthreadMutexAttrState::default()),
            0
        );
        let jobs = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let stopping = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
        let (consumed, receiver) = std::sync::mpsc::channel();
        let workers = (0..3)
            .map(|worker| {
                let jobs = std::sync::Arc::clone(&jobs);
                let stopping = std::sync::Arc::clone(&stopping);
                let consumed = consumed.clone();
                std::thread::spawn(move || {
                    loop {
                        assert_eq!(super::guest_mutex_lock(mutex, false), 0);
                        while jobs.load(std::sync::atomic::Ordering::SeqCst) == 0
                            && !stopping.load(std::sync::atomic::Ordering::SeqCst)
                        {
                            assert_eq!(super::guest_condition_wait(condition, mutex, None), 0);
                        }
                        if stopping.load(std::sync::atomic::Ordering::SeqCst) {
                            assert_eq!(super::guest_mutex_unlock(mutex), 0);
                            break;
                        }
                        assert_eq!(jobs.fetch_sub(1, std::sync::atomic::Ordering::SeqCst), 1);
                        assert_eq!(super::guest_mutex_unlock(mutex), 0);
                        consumed.send(worker).expect("test receiver remains live");
                    }
                })
            })
            .collect::<Vec<_>>();
        drop(consumed);
        wait_for_guest_condition_waiters(condition, 3);

        assert_eq!(super::guest_mutex_lock(mutex, false), 0);
        jobs.store(1, std::sync::atomic::Ordering::SeqCst);
        assert_eq!(super::guest_condition_notify(condition, false), 0);
        assert_eq!(super::guest_mutex_unlock(mutex), 0);
        receiver
            .recv_timeout(std::time::Duration::from_secs(2))
            .expect("one worker consumed the queued task");
        assert!(
            receiver
                .recv_timeout(std::time::Duration::from_millis(50))
                .is_err(),
            "one signal cannot produce a second task consumption"
        );
        wait_for_guest_condition_waiters(condition, 3);

        assert_eq!(super::guest_mutex_lock(mutex, false), 0);
        stopping.store(true, std::sync::atomic::Ordering::SeqCst);
        assert_eq!(super::guest_condition_notify(condition, true), 0);
        assert_eq!(super::guest_mutex_unlock(mutex), 0);
        for worker in workers {
            worker.join().expect("guest worker thread");
        }
        assert_eq!(jobs.load(std::sync::atomic::Ordering::SeqCst), 0);
        assert_eq!(super::guest_mutex_destroy(mutex), 0);
        assert_eq!(super::guest_condition_destroy(condition), 0);
    }

    fn wait_for_guest_condition_waiters(condition: u64, expected: usize) {
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(2);
        loop {
            let waiting = super::guest_sync()
                .0
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner)
                .conditions
                .get(&condition)
                .map_or(0, |condition| condition.waiters.len());
            if waiting == expected {
                return;
            }
            assert!(
                std::time::Instant::now() < deadline,
                "guest condition waiter count did not reach {expected}"
            );
            std::thread::yield_now();
        }
    }

    #[test]
    fn selected_condition_waiter_survives_legal_condition_destruction() {
        let _guard = GUEST_CONDITION_TEST
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let condition = 0x7fff_0000_2100;
        let mutex = 0x7fff_0000_2200;
        assert_eq!(super::guest_condition_init(condition), 0);
        assert_eq!(
            super::guest_mutex_init(mutex, super::PthreadMutexAttrState::default()),
            0
        );
        let (events, receiver) = std::sync::mpsc::channel();
        let waiter = std::thread::spawn(move || {
            assert_eq!(super::guest_mutex_lock(mutex, false), 0);
            events.send("locked").expect("test receiver remains live");
            assert_eq!(super::guest_condition_wait(condition, mutex, None), 0);
            events.send("returned").expect("test receiver remains live");
            assert_eq!(super::guest_mutex_unlock(mutex), 0);
        });
        assert_eq!(
            receiver
                .recv_timeout(std::time::Duration::from_secs(2))
                .expect("waiter acquired guest mutex"),
            "locked"
        );
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(2);
        loop {
            let waiting = super::guest_sync()
                .0
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner)
                .conditions
                .get(&condition)
                .map_or(0, |condition| condition.waiters.len());
            if waiting == 1 {
                break;
            }
            assert!(
                std::time::Instant::now() < deadline,
                "guest waiter did not block"
            );
            std::thread::yield_now();
        }

        assert_eq!(super::guest_mutex_lock(mutex, false), 0);
        assert_eq!(
            super::guest_condition_destroy(condition),
            sce_kernel_error(super::SCE_KERNEL_ERROR_BUSY)
        );
        assert_eq!(super::guest_condition_notify(condition, false), 0);
        assert_eq!(super::guest_condition_destroy(condition), 0);
        assert!(
            receiver
                .recv_timeout(std::time::Duration::from_millis(50))
                .is_err(),
            "selected waiter must reacquire its mutex before returning"
        );
        assert_eq!(super::guest_mutex_unlock(mutex), 0);
        assert_eq!(
            receiver
                .recv_timeout(std::time::Duration::from_secs(2))
                .expect("selected waiter reacquired its mutex"),
            "returned"
        );
        waiter.join().expect("guest waiter thread");
        assert_eq!(super::guest_mutex_destroy(mutex), 0);
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
    fn tls_get_addr_uses_compact_per_thread_module_slots() {
        fn write_template_word(template: &mut [u8], offset: usize, value: u64) {
            template[offset..offset + 8].copy_from_slice(&value.to_le_bytes());
        }

        let template_size =
            usize::try_from(super::GUEST_TLS_PREFIX_SIZE + super::GUEST_TLS_TCB_SIZE)
                .expect("TLS template size fits usize");
        let thread_pointer_offset =
            usize::try_from(super::GUEST_TLS_PREFIX_SIZE).expect("TLS prefix fits usize");
        let dtv_offset = thread_pointer_offset + 0x100;
        let original_thread_pointer = 0x7000_1000_0000_u64;
        let module1_block = original_thread_pointer - 0x40;
        let module2_block = original_thread_pointer - 0x600;
        let mut template = vec![0_u8; template_size];
        write_template_word(
            &mut template,
            thread_pointer_offset,
            original_thread_pointer,
        );
        write_template_word(&mut template, dtv_offset, 1);
        write_template_word(&mut template, dtv_offset + 8, 2);
        assert_eq!(tls_dtv_entry_offset(1), Some(16));
        assert_eq!(tls_dtv_entry_offset(2), Some(24));
        write_template_word(&mut template, dtv_offset + 16, module1_block);
        write_template_word(&mut template, dtv_offset + 24, module2_block);

        let first = clone_guest_tls(&template);
        let second = clone_guest_tls(&template);
        let first_module2 = first.thread_pointer - 0x600;
        let second_module2 = second.thread_pointer - 0x600;
        assert_ne!(first_module2, second_module2);
        assert_eq!(
            read_guest_u64(first.thread_pointer + 0x100 + 24),
            first_module2
        );
        assert_eq!(
            read_guest_u64(second.thread_pointer + 0x100 + 24),
            second_module2
        );
        assert_eq!(read_guest_u64(first.thread_pointer + 0x100 + 32), 0);

        let mut descriptor = [2_u64, 0x280];
        let descriptor_address = descriptor.as_mut_ptr().addr() as u64;
        let mut state = ProbeDispatchState::default();
        set_current_guest_thread_pointer(first.thread_pointer);
        assert_eq!(
            dispatch_probe_import(
                "vNe1w4diLCs",
                [descriptor_address, 0, 0, 0, 0, 0],
                [0; 6],
                &mut state,
            ),
            first_module2 + 0x280
        );
        assert!(state.unimplemented_nids.is_empty());
        assert_eq!(super::known_nid_name("vNe1w4diLCs"), Some("__tls_get_addr"));

        write_guest_u32(first_module2 + 0x280, 0x1122_3344);
        assert_eq!(read_guest_u32(first_module2 + 0x280), 0x1122_3344);
        assert_eq!(read_guest_u32(second_module2 + 0x280), 0);
        set_current_guest_thread_pointer(second.thread_pointer);
        assert_eq!(
            dispatch_probe_import(
                "vNe1w4diLCs",
                [descriptor_address, 0, 0, 0, 0, 0],
                [0; 6],
                &mut state,
            ),
            second_module2 + 0x280
        );

        write_guest_u64(second.thread_pointer + 0x100 + 24, 0);
        assert_eq!(super::guest_tls_get_addr(descriptor_address), 0);
        write_guest_u64(second.thread_pointer + 0x100 + 24, second_module2);
        write_guest_u64(second.thread_pointer + 8, 0);
        assert_eq!(super::guest_tls_get_addr(descriptor_address), 0);
        write_guest_u64(second.thread_pointer + 8, second.thread_pointer + 0x100);

        descriptor[0] = 0;
        assert_eq!(super::guest_tls_get_addr(descriptor_address), 0);
        descriptor[0] = 3;
        assert_eq!(super::guest_tls_get_addr(descriptor_address), 0);
        descriptor[0] = 2;
        descriptor[1] = u64::MAX;
        std::hint::black_box(descriptor[1]);
        assert_eq!(super::guest_tls_get_addr(descriptor_address), 0);
        assert_eq!(super::guest_tls_get_addr(0), 0);
        set_current_guest_thread_pointer(0);
        assert_eq!(super::guest_tls_get_addr(descriptor_address), 0);
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
    fn pthread_once_runs_guest_initializer_exactly_once() {
        ONCE_CALLBACKS.store(0, Ordering::Relaxed);
        let mut control = 0_u32;
        let control_address = (&raw mut control).addr() as u64;
        let routine_address = once_test_callback as *const () as usize as u64;
        std::thread::scope(|scope| {
            for _ in 0..8 {
                scope.spawn(move || {
                    assert_eq!(guest_pthread_once(control_address, routine_address), 0);
                });
            }
        });
        assert_eq!(control, 1);
        assert_eq!(ONCE_CALLBACKS.load(Ordering::Relaxed), 1);
    }

    #[test]
    fn fills_gen5_calendar_time_and_round_trips_timezone_conversions() {
        let before = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs();
        let mut timeval = [0_u64; 2];
        assert_eq!(
            guest_kernel_gettimeofday((&raw mut timeval).addr() as u64),
            0
        );
        assert!((before..=before + 2).contains(&timeval[0]));
        assert!(timeval[1] < 1_000_000);

        let utc = 1_700_000_000_i64;
        let mut local = 0_i64;
        let mut timesec = [0_u8; 16];
        let mut dst = 0_u64;
        assert_eq!(
            guest_kernel_convert_utc_to_localtime([
                u64::from_ne_bytes(utc.to_ne_bytes()),
                (&raw mut local).addr() as u64,
                (&raw mut timesec).addr() as u64,
                (&raw mut dst).addr() as u64,
                0,
                0,
            ]),
            0
        );
        assert_eq!(i64::from_le_bytes(timesec[..8].try_into().unwrap()), utc);
        assert_eq!(
            u64::from(u32::from_le_bytes(timesec[12..16].try_into().unwrap())),
            dst
        );

        let mut round_trip = 0_i64;
        let mut timezone = [0_i32; 2];
        let mut dst_seconds = 0_i32;
        assert_eq!(
            guest_kernel_convert_localtime_to_utc([
                u64::from_ne_bytes(local.to_ne_bytes()),
                0,
                (&raw mut round_trip).addr() as u64,
                (&raw mut timezone).addr() as u64,
                (&raw mut dst_seconds).addr() as u64,
                0,
            ]),
            0
        );
        assert_eq!(round_trip, utc);
        assert_eq!(dst_seconds, i32::try_from(dst).unwrap());
        assert_eq!(timezone[1], dst_seconds / 60);
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
    fn ajm_initialize_allocates_stable_nonzero_context_ids() {
        let mut state = ProbeDispatchState::default();
        let mut first = 0_u32;
        let mut second = 0_u32;
        assert_eq!(
            dispatch_probe_import(
                "dl+4eHSzUu4",
                [0, (&raw mut first).addr() as u64, 0, 0, 0, 0],
                [0; 6],
                &mut state,
            ),
            0
        );
        assert_eq!(
            dispatch_probe_import(
                "dl+4eHSzUu4",
                [0, (&raw mut second).addr() as u64, 0, 0, 0, 0],
                [0; 6],
                &mut state,
            ),
            0
        );
        assert_eq!((first, second), (1, 2));
        assert!(state.ajm_contexts.contains_key(&first));
        assert!(state.ajm_contexts.contains_key(&second));
        assert_eq!(
            super::known_nid_name("dl+4eHSzUu4"),
            Some("sceAjmInitialize")
        );
        assert!(state.unimplemented_nids.is_empty());
    }

    #[test]
    fn np_web_api2_initialize_allocates_bounded_library_contexts() {
        let first =
            super::dispatch_stateless_import("+o9816YQhqQ", [1, 0x8_0000, 0, 0, 0, 0], [0; 6])
                .expect("NP Web API2 initialize is implemented");
        let second =
            super::dispatch_stateless_import("+o9816YQhqQ", [1, 0x8_0000, 0, 0, 0, 0], [0; 6])
                .expect("second NP Web API2 context");

        assert!((1..0x8000).contains(&first));
        assert_eq!(second, first + 1);
        assert_eq!(
            super::known_nid_name("+o9816YQhqQ"),
            Some("sceNpWebApi2Initialize")
        );
    }

    #[test]
    fn ajm_lifecycle_registers_codec_and_instances() {
        let mut state = ProbeDispatchState::default();
        let mut context = 0_u32;
        let mut instance = 0_u32;
        let context_output = (&raw mut context).addr() as u64;
        let instance_output = (&raw mut instance).addr() as u64;

        assert_eq!(
            ajm_initialize([0, context_output, 0, 0, 0, 0], &mut state),
            0
        );
        assert_eq!(
            ajm_module_register([u64::from(context), 1, 0, 0, 0, 0], &mut state),
            0
        );
        assert_eq!(
            ajm_instance_create(
                [u64::from(context), 1, 1, instance_output, 0, 0],
                &mut state,
            ),
            0
        );
        assert_eq!(instance, 0x4001);
        assert_eq!(
            ajm_instance_destroy(
                [u64::from(context), u64::from(instance), 0, 0, 0, 0],
                &mut state,
            ),
            0
        );
        assert_eq!(
            ajm_module_unregister([u64::from(context), 1, 0, 0, 0, 0], &mut state),
            0
        );
        assert_eq!(
            ajm_finalize([u64::from(context), 0, 0, 0, 0, 0], &mut state),
            0
        );
        assert!(state.ajm_contexts.is_empty());
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
    fn pending_guest_exception_runs_on_target_import_with_orbis_stack_context() {
        let _guard = GUEST_EXCEPTION_TEST
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        const THREAD: u64 = 0x1234;
        const GUEST_RSP: u64 = 0x7000_1234_5678;
        DELIVERED_EXCEPTION.store(0, Ordering::SeqCst);
        DELIVERED_EXCEPTION_RSP.store(0, Ordering::SeqCst);
        super::set_current_guest_thread_handle(THREAD);
        let handler = capture_guest_exception as *const () as usize as u64;
        let mut delivery = super::guest_exception_delivery()
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        *delivery = super::GuestExceptionDeliveryState::default();
        delivery.pending.insert(
            THREAD,
            super::PendingGuestException {
                handler,
                exception: 30,
            },
        );
        drop(delivery);

        super::deliver_pending_guest_exception_at_import(&super::ImportCallFrame {
            guest_rsp: GUEST_RSP,
            guest_return_address: 0x8000_1234,
            ..super::ImportCallFrame::default()
        });

        assert_eq!(DELIVERED_EXCEPTION.load(Ordering::SeqCst), 30);
        assert_eq!(
            DELIVERED_EXCEPTION_RSP.load(Ordering::SeqCst),
            usize::try_from(GUEST_RSP).unwrap()
        );
        let delivery = super::guest_exception_delivery()
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        assert!(delivery.pending.is_empty());
        assert!(delivery.active.is_empty());
        drop(delivery);
        super::set_current_guest_thread_handle(0);
    }

    #[test]
    fn pending_guest_exception_interrupts_event_flag_and_semaphore_waits() {
        let _guard = GUEST_EXCEPTION_TEST
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        const THREAD: u64 = 0x5678;
        const EVENT_FLAG: u64 = 0x7fff_0000_4000;
        const SEMAPHORE: u32 = 0x7fff_4000;
        super::set_current_guest_thread_handle(THREAD);
        super::guest_sync()
            .0
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .event_flags
            .insert(
                EVENT_FLAG,
                super::GuestEventFlagState {
                    name: "exception-test".to_owned(),
                    attributes: 0,
                    bits: 0,
                    waiters: 0,
                    cancellation_generation: 0,
                },
            );
        super::guest_sync()
            .0
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .semaphores
            .insert(
                SEMAPHORE,
                super::GuestSemaphoreState {
                    count: 0,
                    maximum: 1,
                },
            );
        super::guest_exception_delivery()
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .pending
            .insert(
                THREAD,
                super::PendingGuestException {
                    handler: 1,
                    exception: 30,
                },
            );

        assert_eq!(
            super::guest_event_flag_wait([
                EVENT_FLAG,
                1,
                super::EVENT_FLAG_WAIT_OR as u64,
                0,
                0,
                0
            ]),
            sce_kernel_error(super::SCE_KERNEL_ERROR_INTERRUPTED)
        );
        assert_eq!(
            super::guest_semaphore_wait([u64::from(SEMAPHORE), 1, 0, 0, 0, 0]),
            sce_kernel_error(super::SCE_KERNEL_ERROR_INTERRUPTED)
        );

        let sync = super::guest_sync();
        let mut state = sync
            .0
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        assert_eq!(state.event_flags[&EVENT_FLAG].waiters, 0);
        state.event_flags.remove(&EVENT_FLAG);
        state.semaphores.remove(&SEMAPHORE);
        drop(state);
        super::guest_exception_delivery()
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .pending
            .remove(&THREAD);
        super::set_current_guest_thread_handle(0);
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
        assert_eq!(
            pthread_attr_set_affinity([attribute_address, 0x7f, 0, 0, 0, 0], &mut state),
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
            0x7f
        );
        assert_eq!(affinity_mask, 0x7f);
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
        let mut files = KERNEL_FILES
            .get_or_init(|| Mutex::new(KernelFileState::default()))
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        *files = KernelFileState::default();
        files.files.insert(3, file);
        drop(files);

        assert_eq!(guest_kernel_lseek([3, 2, 0, 0, 0, 0]), 2);
        let mut bytes = [0_u8; 2];
        assert_eq!(
            guest_kernel_read([3, bytes.as_mut_ptr().addr() as u64, 2, 0, 0, 0]),
            2
        );
        assert_eq!(&bytes, b"cd");

        let mut stat = [0_u8; super::SCE_KERNEL_STAT_SIZE];
        assert_eq!(
            guest_kernel_fstat([3, stat.as_mut_ptr().addr() as u64, 0, 0, 0, 0]),
            0
        );
        assert_eq!(i64::from_le_bytes(stat[72..80].try_into().unwrap()), 6);
        assert_ne!(u32::from_le_bytes(stat[4..8].try_into().unwrap()), 0);
        assert_ne!(u16::from_le_bytes(stat[8..10].try_into().unwrap()), 0);
        assert_ne!(u32::from_le_bytes(stat[88..92].try_into().unwrap()), 0);

        assert_eq!(guest_kernel_ftruncate([3, 4, 0, 0, 0, 0]), 0);
        assert_eq!(
            KERNEL_FILES
                .get()
                .unwrap()
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner)
                .files
                .get(&3)
                .unwrap()
                .metadata()
                .unwrap()
                .len(),
            4
        );
    }

    /// Registering interest must not announce a completion: the guest's listener resolves the
    /// event against state the submission creates, so an early wake dereferences a context that
    /// does not exist.
    #[test]
    fn ampr_completions_are_announced_by_submission_not_registration() {
        let mut state = ProbeDispatchState::default();
        let equeue = 0x4000;
        state.kernel_equeues.insert(equeue);
        assert_eq!(
            kernel_add_ampr_event([equeue, 0x7502, 0xabc, 0, 0, 0], &mut state),
            0
        );
        assert!(
            state.pending_events.is_empty(),
            "registration announced a completion"
        );

        let mut buffer = [0_u8; 0x100];
        let command_buffer = 0x5000_u64;
        state.ampr_command_buffers.insert(
            command_buffer,
            AmprCommandBufferState {
                buffer: buffer.as_mut_ptr().addr() as u64,
                size: buffer.len() as u64,
                ..AmprCommandBufferState::default()
            },
        );
        assert_eq!(
            ampr_command_buffer_write_kernel_event_queue(
                [command_buffer, equeue, 0x7502, 0xabc, 0, 0],
                &mut state,
            ),
            0
        );
        assert!(
            state.pending_events.is_empty(),
            "recording announced a completion before submission"
        );

        assert_eq!(
            ampr_submit_command_buffer([command_buffer, 0, 0, 0, 0, 0], &mut state),
            0
        );
        assert_eq!(state.pending_events.len(), 1);
        let event = state.pending_events.front().unwrap();
        assert_eq!(event.equeue, equeue);
        assert_eq!(event.ident, 0x7502);
        assert_eq!(event.user_data, 0xabc);
    }

    #[test]
    fn directory_records_pack_console_dirents_without_crossing_blocks() {
        let entries: Vec<(String, bool)> = (0..40)
            .map(|index| (format!("pakchunk{index}-ps5.pak"), index % 7 == 0))
            .collect();
        let mut next_file_number = 1;
        let directory = KernelDirectory::new(&entries, &mut next_file_number);
        assert_eq!(directory.records.len() % DIRENT_BLOCK_SIZE, 0);

        // Walk the records exactly as the guest does and recover every name in order.
        let mut names = Vec::new();
        let mut offset = 0;
        while offset < directory.records.len() {
            let record = &directory.records[offset..];
            let reclen = usize::from(u16::from_le_bytes([record[4], record[5]]));
            let namlen = usize::from(record[7]);
            if namlen == 0 {
                offset += DIRENT_BLOCK_SIZE - offset % DIRENT_BLOCK_SIZE;
                continue;
            }
            assert!(reclen >= DIRENT_HEADER_SIZE + namlen + 1);
            // A record must lie entirely inside one 512-byte block.
            assert_eq!(
                offset / DIRENT_BLOCK_SIZE,
                (offset + DIRENT_HEADER_SIZE + namlen) / DIRENT_BLOCK_SIZE,
                "record at {offset} crossed a block boundary"
            );
            names.push(String::from_utf8_lossy(&record[8..8 + namlen]).into_owned());
            assert_eq!(record[6], if names.len() % 7 == 1 { 4 } else { 8 });
            offset += reclen;
        }
        assert_eq!(
            names,
            entries
                .iter()
                .map(|(name, _)| name.clone())
                .collect::<Vec<_>>()
        );
        assert_eq!(next_file_number, 41);
    }

    /// Engines store timestamps as ticks and decompose them again for display. A tick that does
    /// not survive the round trip produces an implausible date and fails their validation.
    #[test]
    fn rtc_tick_round_trips_through_a_date() {
        let mut date = [0_u16; 8];
        date[0] = 2026;
        date[1] = 7;
        date[2] = 25;
        date[3] = 18;
        date[4] = 43;
        date[5] = 7;
        let mut tick = 0_u64;
        assert_eq!(
            rtc_get_tick([
                date.as_mut_ptr().addr() as u64,
                (&raw mut tick).addr() as u64,
                0,
                0,
                0,
                0,
            ]),
            0
        );
        assert_ne!(tick, 0);

        let mut recovered = [0_u16; 8];
        assert_eq!(
            rtc_set_tick([
                recovered.as_mut_ptr().addr() as u64,
                (&raw const tick).addr() as u64,
                0,
                0,
                0,
                0,
            ]),
            0
        );
        assert_eq!(recovered[..6], date[..6]);

        // The current clock must also decompose to the same instant it reports as a tick.
        let mut now_tick = 0_u64;
        assert_eq!(rtc_get_current_tick((&raw mut now_tick).addr() as u64), 0);
        let mut now_date = [0_u16; 8];
        assert_eq!(
            rtc_set_tick([
                now_date.as_mut_ptr().addr() as u64,
                (&raw const now_tick).addr() as u64,
                0,
                0,
                0,
                0,
            ]),
            0
        );
        assert!(now_date[0] >= 2020, "year was {}", now_date[0]);
        assert!((1..=12).contains(&now_date[1]));
    }

    #[test]
    fn civil_days_round_trip_through_civil_decomposition() {
        for seconds in [
            0_i64,
            1_709_164_800,
            951_782_400,
            -2_203_891_201,
            1_800_000_000,
        ] {
            let (year, month, day, hour, minute, second) = civil_from_unix_seconds(seconds);
            let days = unix_days_from_civil(i64::from(year), i64::from(month), i64::from(day));
            let rebuilt = days * 86_400
                + i64::from(hour) * 3_600
                + i64::from(minute) * 60
                + i64::from(second);
            assert_eq!(rebuilt, seconds, "round trip failed for {seconds}");
        }
    }

    #[test]
    fn civil_decomposition_matches_known_timestamps() {
        // Epoch, a leap day, the end of a non-leap century year, and a pre-epoch date.
        assert_eq!(civil_from_unix_seconds(0), (1970, 1, 1, 0, 0, 0));
        assert_eq!(
            civil_from_unix_seconds(1_709_164_800),
            (2024, 2, 29, 0, 0, 0)
        );
        // Year 2000 is a leap year despite being a century, so the day after February 28 exists.
        assert_eq!(
            civil_from_unix_seconds(951_782_399),
            (2000, 2, 28, 23, 59, 59)
        );
        assert_eq!(civil_from_unix_seconds(951_782_400), (2000, 2, 29, 0, 0, 0));
        // Year 1900 is not a leap year, so February ends after the 28th.
        assert_eq!(
            civil_from_unix_seconds(-2_203_891_201),
            (1900, 2, 28, 23, 59, 59)
        );
        assert_eq!(civil_from_unix_seconds(-1), (1969, 12, 31, 23, 59, 59));
    }

    #[test]
    fn current_local_time_is_a_valid_date_and_rejects_a_null_output() {
        let mut time = [0_u16; 8];
        assert_eq!(
            rtc_get_current_clock_local_time(time.as_mut_ptr().addr() as u64),
            0
        );
        // Engines abort on an all-zero date, so every field must be in its calendar range.
        assert!(time[0] >= 2020, "year was {}", time[0]);
        assert!((1..=12).contains(&time[1]), "month was {}", time[1]);
        assert!((1..=31).contains(&time[2]), "day was {}", time[2]);
        assert!(time[3] < 24 && time[4] < 60 && time[5] < 60);
        assert_ne!(rtc_get_current_clock_local_time(0), 0);
    }

    /// Builds one `sceKernelBatchMap` entry: start, direct offset, length, protection, type, and
    /// operation at the console's fixed field offsets.
    #[cfg(target_os = "macos")]
    fn batch_map_entry(
        start: u64,
        offset: u64,
        length: u64,
        protection: u8,
        memory_type: u8,
        operation: u32,
    ) -> [u8; 32] {
        let mut entry = [0_u8; 32];
        entry[0..8].copy_from_slice(&start.to_le_bytes());
        entry[8..16].copy_from_slice(&offset.to_le_bytes());
        entry[16..24].copy_from_slice(&length.to_le_bytes());
        entry[24] = protection;
        entry[25] = memory_type;
        entry[28..32].copy_from_slice(&operation.to_le_bytes());
        entry
    }

    #[cfg(target_os = "macos")]
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

    #[cfg(target_os = "macos")]
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

    #[cfg(target_os = "macos")]
    #[test]
    fn maps_named_flexible_memory_with_cpu_read_write_protection() {
        let mut state = ProbeDispatchState::default();
        let mut mapped = 0_u64;
        let name = b"CppWebApi\0";
        assert_eq!(
            super::map_named_flexible_memory(
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
        assert_eq!(output[32], 0x11);
        assert_eq!(&output[33..41], b"flexible");
        assert_eq!(
            super::known_nid_name("mL8NDH86iQI"),
            Some("sceKernelMapNamedFlexibleMemory")
        );
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn rejects_invalid_named_flexible_memory_arguments() {
        let mut state = ProbeDispatchState::default();
        let mut mapped = 0_u64;
        let short_name = b"flex\0";
        let long_name = [b'x'; 33];
        let mut terminated_long_name = [0_u8; 34];
        terminated_long_name[..33].copy_from_slice(&long_name);

        assert_eq!(
            super::map_named_flexible_memory(
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
            sce_kernel_error(super::SCE_KERNEL_ERROR_INVALID_ARGUMENT)
        );
        assert_eq!(
            super::map_named_flexible_memory(
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
            sce_kernel_error(super::SCE_KERNEL_ERROR_NAME_TOO_LONG)
        );
        assert_eq!(
            super::map_named_flexible_memory(
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
            sce_kernel_error(super::SCE_KERNEL_ERROR_INVALID_ARGUMENT)
        );
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn mspace_allocates_inside_named_flexible_memory_and_reuses_freed_blocks() {
        let mut state = ProbeDispatchState::default();
        let mut mapped = 0_u64;
        let mapping_name = b"WebApiBacking\0";
        assert_eq!(
            super::map_named_flexible_memory(
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
        let handle = super::guest_mspace_create(
            [mspace_name.as_ptr().addr() as u64, mapped, 0x8000, 0, 0, 0],
            &mut state,
        );
        assert_eq!(handle, mapped);

        let first = super::guest_mspace_malloc([handle, 0x58, 0, 0, 0, 0], &mut state);
        let second = super::guest_mspace_malloc([handle, 0x30, 0, 0, 0, 0], &mut state);
        assert!(first >= mapped + 0x100);
        assert_eq!(second, first + 0x60);
        // SAFETY: `first` is a live mspace allocation of 0x60 usable bytes.
        unsafe { std::ptr::write_bytes(first as *mut u8, 0x5a, 0x58) };
        let grown = super::guest_mspace_realloc([handle, first, 0x100, 0, 0, 0], &mut state);
        assert_ne!(grown, 0);
        // SAFETY: realloc preserved the first 0x58 bytes in the returned live allocation.
        assert!(
            unsafe { std::slice::from_raw_parts(grown as *const u8, 0x58) }
                .iter()
                .all(|byte| *byte == 0x5a)
        );
        assert_eq!(
            super::guest_mspace_usable_size([handle, grown, 0, 0, 0, 0], &state),
            0x100
        );

        super::guest_mspace_free([handle, second, 0, 0, 0, 0], &mut state);
        super::guest_mspace_free([handle, grown, 0, 0, 0, 0], &mut state);
        assert_eq!(
            super::guest_mspace_is_heap_empty([handle, 0, 0, 0, 0, 0], &state),
            1
        );
        assert_eq!(
            super::guest_mspace_malloc([handle, 0x58, 0, 0, 0, 0], &mut state),
            first
        );
        assert_eq!(
            super::known_nid_name("OJjm-QOIHlI"),
            Some("sceLibcMspaceMalloc")
        );
    }

    #[test]
    fn json_initializer_tracks_allocator_and_file_buffer_configuration() {
        let mut state = ProbeDispatchState::default();
        let mut initializer = [0_u8; 8];
        let mut parameter = [0_u8; 0x28];
        let mut allocator = [0_u8; 8];
        let initializer_address = initializer.as_mut_ptr().addr() as u64;
        let parameter_address = parameter.as_mut_ptr().addr() as u64;
        let allocator_address = allocator.as_mut_ptr().addr() as u64;

        assert_eq!(
            super::json_init_parameter_set_allocator(
                [parameter_address, allocator_address, 0x1234, 0, 0, 0,],
                &mut state,
            ),
            0
        );
        assert_eq!(
            super::json_init_parameter_set_file_buffer_size(
                [parameter_address, 0x200, 0, 0, 0, 0],
                &mut state,
            ),
            0
        );
        assert_eq!(
            super::json_initializer_initialize(
                [initializer_address, parameter_address, 0, 0, 0, 0],
                &mut state,
            ),
            0
        );
        assert_eq!(
            state.json_runtime.allocators.get(&parameter_address),
            Some(&(allocator_address, 0x1234))
        );
        assert_eq!(
            state.json_runtime.file_buffer_sizes.get(&parameter_address),
            Some(&0x200)
        );
        assert_eq!(
            state
                .json_runtime
                .initialized_objects
                .get(&initializer_address),
            Some(&parameter_address)
        );
        assert_eq!(
            super::known_nid_name("IXW-z8pggfg"),
            Some("sce::Json::Initializer::initialize(InitParameter2)")
        );
        assert_eq!(
            super::json_initializer_terminate([initializer_address, 0, 0, 0, 0, 0], &mut state,),
            0
        );
    }

    #[test]
    fn offline_np_service_initializers_retain_context_and_property_ownership() {
        let mut state = ProbeDispatchState::default();
        assert_ne!(super::sysmodule_is_loaded(0xcc, &state), 0);
        assert_eq!(super::sysmodule_load(0xcc, &mut state), 0);
        assert_eq!(super::sysmodule_is_loaded(0xcc, &state), 0);
        assert_eq!(super::game_update_initialize(&mut state), 0);

        let entitlement_parameters = [0_u8; 0x20];
        let mut entitlement_context = [0_u8; 0x20];
        let entitlement_context_address = entitlement_context.as_mut_ptr().addr() as u64;
        let entitlement_parameter_address = entitlement_parameters.as_ptr().addr() as u64;
        assert_eq!(
            super::np_entitlement_access_initialize(
                [
                    entitlement_context_address,
                    entitlement_parameter_address,
                    0,
                    0,
                    0,
                    0,
                ],
                &mut state,
            ),
            0
        );
        let signaling_parameters = [0x5a_u8; 0x28];
        assert_eq!(
            super::np_session_signaling_initialize(
                [signaling_parameters.as_ptr().addr() as u64, 0, 0, 0, 0, 0],
                &mut state,
            ),
            0
        );
        let user_context =
            super::np_web_api2_create_user_context([1, 0x1000_0000, 0, 0, 0, 0], &mut state);
        assert_eq!(user_context, 1);
        assert_eq!(super::np_check_callback(&mut state), 0);
        assert_eq!(state.services.np_callback_polls, 1);

        let mut universal_parameters = [0_u8; 16];
        universal_parameters[..8].copy_from_slice(&16_u64.to_ne_bytes());
        assert_eq!(
            super::np_universal_data_initialize(
                universal_parameters.as_ptr().addr() as u64,
                &mut state,
            ),
            0
        );
        let mut object = 0_u64;
        assert_eq!(
            super::np_universal_data_create_property_object(
                (&raw mut object).addr() as u64,
                &mut state,
            ),
            0
        );
        assert_ne!(object, 0);
        assert_eq!(
            super::known_nid_name("s6W4Zl4Slgk"),
            Some("sceNpUniversalDataSystemCreateEventPropertyObject")
        );
    }

    #[test]
    fn av_player_initialization_accepts_the_gen5_layout_and_owns_handles() {
        let mut state = ProbeDispatchState::default();
        let mut initialization = [0_u8; 0x230];
        let initialization_size = initialization.len() as u64;
        initialization[..8].copy_from_slice(&initialization_size.to_ne_bytes());
        for offset in [16_usize, 24, 32, 40] {
            initialization[offset..offset + 8]
                .copy_from_slice(&(0x1000 + offset as u64).to_ne_bytes());
        }
        let mut handle = 0_u64;
        assert_eq!(
            super::av_player_set_log_callback([0x1234, 0x5678, 0, 0, 0, 0], &mut state),
            0
        );
        assert_eq!(
            super::av_player_init_ex(
                [
                    initialization.as_ptr().addr() as u64,
                    (&raw mut handle).addr() as u64,
                    0,
                    0,
                    0,
                    0,
                ],
                &mut state,
            ),
            0
        );
        assert_eq!(handle, 1);
        assert_eq!(
            state.av_player.players[&handle]._initialization.len(),
            initialization.len()
        );
        let post_initialization = 1_u64;
        assert_eq!(
            super::av_player_post_init(
                [
                    handle,
                    (&raw const post_initialization).addr() as u64,
                    0,
                    0,
                    0,
                    0,
                ],
                &state,
            ),
            0
        );
        let source = b"/app0/Media/intro.mp4\0";
        assert_eq!(
            super::av_player_add_source(
                [handle, source.as_ptr().addr() as u64, 0, 0, 0, 0],
                &mut state,
            ),
            sce_kernel_error(super::AV_PLAYER_ERROR_OPERATION_FAILED)
        );
        assert_eq!(super::av_player_is_active(handle, &state), 0);
        assert_eq!(super::av_player_close(handle, &mut state), 0);
        assert_eq!(
            super::known_nid_name("o9eWRkSL+M4"),
            Some("sceAvPlayerInitEx")
        );
        assert_eq!(
            super::known_nid_name("HD1YKVU26-M"),
            Some("sceAvPlayerPostInit")
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
    fn emits_ordered_gen5_release_and_wait_label_operations() {
        let _event_guard = GPU_EVENT_SINK_TEST
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let mut label = 0_u64;
        let label_address = (&raw mut label).addr() as u64;
        let commands = [
            agc_pm4(8, AGC_IT_NOP, super::AGC_R_RELEASE_MEM),
            0,
            2 << 16,
            low_u32(label_address),
            low_u32(label_address >> 32),
            0x89ab_cdef,
            0x0123_4567,
            0,
            agc_pm4(9, AGC_IT_NOP, super::AGC_R_WAIT_MEM_64),
            low_u32(label_address),
            low_u32(label_address >> 32),
            u32::MAX,
            u32::MAX,
            0x89ab_cdef,
            0x0123_4567,
            3,
            0,
        ];
        let (sender, receiver) = remu_gpu::gpu_event_channel();
        remu_gpu::install_event_sink(sender);
        let mut graphics = AgcGraphicsState::default();
        let mut shaders = AgcShaderRegistry::default();
        let mut submission_count = 0;
        super::agc_parse_submission(
            commands.as_ptr().addr() as u64,
            u32::try_from(commands.len()).expect("command dword count"),
            &mut super::AgcSubmissionContext {
                graphics: &mut graphics,
                shaders: &mut shaders,
                index_resources: &BTreeMap::new(),
                video_out: &VideoOutState::default(),
                submission_count: &mut submission_count,
                flips: &mut Vec::new(),
                event_writes: &mut Vec::new(),
            },
        )
        .expect("release/wait submission");
        remu_gpu::clear_event_sink();

        assert_eq!(
            receiver.recv().expect("release event"),
            remu_gpu::GpuEvent::MemoryRelease(remu_gpu::GpuMemoryRelease {
                address: label_address,
                value: remu_gpu::GpuMemoryReleaseValue::Immediate64(0x0123_4567_89ab_cdef),
            })
        );
        assert_eq!(
            receiver.recv().expect("wait event"),
            remu_gpu::GpuEvent::MemoryWait(remu_gpu::GpuMemoryWait {
                address: label_address,
                reference: 0x0123_4567_89ab_cdef,
                mask: u64::MAX,
                compare_function: 3,
                is_64_bit: true,
            })
        );
        assert_eq!(submission_count, 1);
        assert!(receiver.try_recv().is_err());
    }

    #[test]
    fn emits_ordered_standard_and_agc_write_data_operations() {
        let _event_guard = GPU_EVENT_SINK_TEST
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let mut labels = [0_u32; 4];
        let label_address = labels.as_mut_ptr().addr() as u64;
        let commands = [
            agc_pm4(5, super::AGC_IT_WRITE_DATA, 0),
            5 << 8,
            low_u32(label_address),
            low_u32(label_address >> 32),
            0x1122_3344,
            agc_pm4(6, AGC_IT_NOP, super::AGC_R_WRITE_DATA),
            4 | (1 << 16),
            low_u32(label_address + 8),
            low_u32((label_address + 8) >> 32),
            0x5566_7788,
            0x99aa_bbcc,
        ];
        let (sender, receiver) = remu_gpu::gpu_event_channel();
        remu_gpu::install_event_sink(sender);
        let mut graphics = AgcGraphicsState::default();
        let mut shaders = AgcShaderRegistry::default();
        let mut submission_count = 0;
        super::agc_parse_submission(
            commands.as_ptr().addr() as u64,
            u32::try_from(commands.len()).expect("command dword count"),
            &mut super::AgcSubmissionContext {
                graphics: &mut graphics,
                shaders: &mut shaders,
                index_resources: &BTreeMap::new(),
                video_out: &VideoOutState::default(),
                submission_count: &mut submission_count,
                flips: &mut Vec::new(),
                event_writes: &mut Vec::new(),
            },
        )
        .expect("write-data submission");
        remu_gpu::clear_event_sink();

        assert_eq!(
            receiver.recv().expect("standard write event"),
            remu_gpu::GpuEvent::MemoryWrite(remu_gpu::GpuMemoryWrite {
                address: label_address,
                values: vec![0x1122_3344],
                increment_address: true,
            })
        );
        assert_eq!(
            receiver.recv().expect("AGC write event"),
            remu_gpu::GpuEvent::MemoryWrite(remu_gpu::GpuMemoryWrite {
                address: label_address + 8,
                values: vec![0x5566_7788, 0x99aa_bbcc],
                increment_address: false,
            })
        );
        assert_eq!(submission_count, 1);
        assert!(receiver.try_recv().is_err());
    }

    #[test]
    fn decodes_partial_flush_events_without_inventing_completion_notifications() {
        for (event_type, wait_for) in [
            (0x07, remu_gpu::GpuPipelineStage::ComputeShader),
            (0x0f, remu_gpu::GpuPipelineStage::VertexShader),
            (0x10, remu_gpu::GpuPipelineStage::PixelShader),
        ] {
            let packet = [agc_pm4(2, super::AGC_IT_EVENT_WRITE, 0), 0x400 | event_type];
            assert_eq!(
                super::decode_agc_pipeline_barrier(packet.as_ptr().addr() as u64, 2)
                    .expect("supported partial flush"),
                remu_gpu::GpuPipelineBarrier { wait_for }
            );
        }

        let wrong_index = [agc_pm4(2, super::AGC_IT_EVENT_WRITE, 0), 0x10];
        assert!(matches!(
            super::decode_agc_pipeline_barrier(wrong_index.as_ptr().addr() as u64, 2),
            Err(super::AgcSubmissionError::PartialFlushEventIndex {
                event_type: 0x10,
                actual: 0,
            })
        ));
        let unsupported = [agc_pm4(2, super::AGC_IT_EVENT_WRITE, 0), 0x416];
        assert!(matches!(
            super::decode_agc_pipeline_barrier(unsupported.as_ptr().addr() as u64, 2),
            Err(super::AgcSubmissionError::UnsupportedEventWrite {
                event_type: 0x16,
                event_index: 4,
                length: 2,
            })
        ));
        let reserved = [agc_pm4(2, super::AGC_IT_EVENT_WRITE, 0), 0x8000_0410];
        assert!(matches!(
            super::decode_agc_pipeline_barrier(reserved.as_ptr().addr() as u64, 2),
            Err(super::AgcSubmissionError::UnsupportedEventWriteControl(
                0x8000_0410
            ))
        ));
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

        super::agc_handle_submission_packet(
            super::AgcSubmissionPacket {
                address: packet.as_ptr().addr() as u64,
                length: 2,
                opcode: super::AGC_IT_CLEAR_STATE,
                register: 0,
                predicated: false,
            },
            &mut graphics,
            &mut AgcShaderRegistry::default(),
            &BTreeMap::new(),
            &VideoOutState::default(),
            &mut Vec::new(),
            &mut Vec::new(),
            None,
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
        let mut event_writes = Vec::new();

        let mut invoke = |address, length, graphics: &mut AgcGraphicsState| {
            super::agc_handle_submission_packet(
                super::AgcSubmissionPacket {
                    address,
                    length,
                    opcode: super::AGC_IT_CLEAR_STATE,
                    register: 0,
                    predicated: false,
                },
                graphics,
                &mut shaders,
                &index_resources,
                &video_out,
                &mut flips,
                &mut event_writes,
                None,
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
    fn distinguishes_absent_disabled_programmed_and_malformed_color_target_state() {
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
            AgcColorTargetState::Disabled
        ));
    }

    #[test]
    fn explicit_zero_target_mask_does_not_enable_targetless_scanout_fallback() {
        let target = super::AgcVideoOutTarget {
            address: 0x0070_1234_5600,
            attributes: VideoOutBufferAttributes {
                width: 1_280,
                height: 720,
                pitch: 1_280,
                pixel_format: 0x8000_0000_2200_0000,
                tiling_mode: 0,
            },
        };
        let mut graphics = AgcGraphicsState::default();
        graphics
            .context_registers
            .insert(super::AGC_CB_TARGET_MASK, 0);
        let (_, color_target_absent) = super::agc_prepare_draw_color_targets(&mut graphics);
        assert!(!color_target_absent);

        let mut draw = targetless_scanout_test_draw();
        draw.pixel_shader_evaluation = Some(Arc::new(remu_shader::ShaderEvaluation {
            initial_scalar_registers: Vec::new().into(),
            scalar_registers: Vec::new().into(),
            scalar_memory_reads: Vec::new(),
            image_bindings: Vec::new(),
            global_memory_bindings: Vec::new(),
            compute_system_registers: None,
            vertex_inputs: Vec::new(),
        }));
        super::agc_resolve_targetless_scanout(
            &mut graphics,
            &mut draw,
            color_target_absent,
            Some(target),
        );
        assert_eq!(draw.scanout_target, None);
    }

    #[test]
    fn scanout_lookahead_is_submission_bounded_and_not_limited_by_channel_capacity() {
        let address = 0x0070_1234_5600_u64;
        let video_out = VideoOutState {
            is_open: true,
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
        let flip = [agc_pm4(6, AGC_IT_NOP, super::AGC_R_FLIP), 1, 0, 0, 0, 0];
        let draw_packet = || super::AgcSubmissionPacket {
            address: 0,
            length: 3,
            opcode: super::AGC_IT_DRAW_INDEX_AUTO,
            register: 0,
            predicated: false,
        };
        let mut packets = vec![draw_packet()];
        packets.extend((0..(remu_gpu::GPU_EVENT_QUEUE_CAPACITY + 17)).map(|_| {
            super::AgcSubmissionPacket {
                address: 0,
                length: 2,
                opcode: AGC_IT_NOP,
                register: 0,
                predicated: false,
            }
        }));
        let second_draw = packets.len();
        packets.push(draw_packet());
        packets.push(super::AgcSubmissionPacket {
            address: flip.as_ptr().addr() as u64,
            length: 6,
            opcode: AGC_IT_NOP,
            register: super::AGC_R_FLIP,
            predicated: false,
        });

        let future = super::agc_future_scanout_targets(&packets, &video_out);
        assert_eq!(future[0].map(|target| target.address), Some(address));
        assert_eq!(
            future[second_draw].map(|target| target.address),
            Some(address)
        );

        let boundary_packets = [
            draw_packet(),
            super::AgcSubmissionPacket {
                address: 0,
                length: 2,
                opcode: super::AGC_IT_CLEAR_STATE,
                register: 0,
                predicated: false,
            },
            draw_packet(),
            *packets.last().expect("flip packet"),
        ];
        let boundary_future = super::agc_future_scanout_targets(&boundary_packets, &video_out);
        assert_eq!(boundary_future[0], None);
        assert_eq!(
            boundary_future[2].map(|target| target.address),
            Some(address)
        );
    }

    #[test]
    fn executes_native_draw_index_2_with_embedded_index_base_and_count() {
        let _event_guard = GPU_EVENT_SINK_TEST
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let mut indices = [0_u16, 1, 2, 2, 3, 0];
        let index_address = indices.as_ptr().addr() as u64;
        let index_resource = 0x0070_1234_5600_u64;
        let commands = [
            agc_pm4(6, super::AGC_IT_DRAW_INDEX_2, 0),
            6,
            low_u32(index_address),
            low_u32(index_address >> 32),
            6,
            2,
        ];
        let (sender, receiver) = remu_gpu::gpu_event_channel();
        remu_gpu::install_event_sink(sender);
        let mut graphics = AgcGraphicsState::default();
        let mut shaders = AgcShaderRegistry::default();
        let mut submission_count = 0;

        super::agc_parse_submission(
            commands.as_ptr().addr() as u64,
            u32::try_from(commands.len()).expect("command dword count"),
            &mut super::AgcSubmissionContext {
                graphics: &mut graphics,
                shaders: &mut shaders,
                index_resources: &BTreeMap::from([(index_address, index_resource)]),
                video_out: &VideoOutState::default(),
                submission_count: &mut submission_count,
                flips: &mut Vec::new(),
                event_writes: &mut Vec::new(),
            },
        )
        .expect("native DRAW_INDEX_2 submission");
        remu_gpu::clear_event_sink();

        let remu_gpu::GpuEvent::IndexedDraw(draw) = receiver.recv().expect("indexed draw event")
        else {
            panic!("DRAW_INDEX_2 emitted a non-draw event");
        };
        assert!(draw.indexed);
        assert_eq!(draw.index_buffer_address, index_address);
        assert_eq!(draw.index_resource_address, index_resource);
        assert_eq!(draw.index_buffer_count, 6);
        assert_eq!(draw.index_offset, 0);
        assert_eq!(draw.index_count, 6);
        indices.fill(u16::MAX);
        assert_eq!(
            draw.index_snapshot.as_deref(),
            Some([0, 1, 2, 2, 3, 0].as_slice())
        );
        assert_eq!(draw.draw_flags, 2);
        assert_eq!(submission_count, 1);
        assert!(receiver.try_recv().is_err());
    }

    #[test]
    fn targetless_scanout_resolution_requires_complete_unambiguous_draw_state() {
        let target = super::AgcVideoOutTarget {
            address: 0x0070_1234_5600,
            attributes: VideoOutBufferAttributes {
                width: 1_280,
                height: 720,
                pitch: 1_280,
                pixel_format: 0x8000_0000_2200_0000,
                tiling_mode: 0,
            },
        };
        let mut graphics = AgcGraphicsState::default();
        let mut draw = targetless_scanout_test_draw();

        super::agc_resolve_targetless_scanout(&mut graphics, &mut draw, true, Some(target));
        assert_eq!(
            draw.scanout_target, None,
            "failed resource evaluation is not guessed"
        );

        draw.pixel_shader_evaluation = Some(Arc::new(remu_shader::ShaderEvaluation {
            initial_scalar_registers: Vec::new().into(),
            scalar_registers: Vec::new().into(),
            scalar_memory_reads: Vec::new(),
            image_bindings: Vec::new(),
            global_memory_bindings: Vec::new(),
            compute_system_registers: None,
            vertex_inputs: Vec::new(),
        }));
        super::agc_resolve_targetless_scanout(&mut graphics, &mut draw, true, Some(target));
        assert_eq!(
            draw.scanout_target.map(|scanout| scanout.address),
            Some(target.address)
        );

        draw.scanout_target = None;
        graphics
            .context_registers
            .insert(super::AGC_DB_DEPTH_CONTROL, 0x2);
        super::agc_resolve_targetless_scanout(&mut graphics, &mut draw, true, Some(target));
        assert_eq!(
            draw.scanout_target.map(|scanout| scanout.address),
            Some(target.address),
            "stale depth control without a bound depth surface does not suppress scanout"
        );

        draw.scanout_target = None;
        graphics.context_registers.insert(super::AGC_DB_Z_INFO, 3);
        graphics
            .context_registers
            .insert(super::AGC_DB_DEPTH_SIZE_XY, 1_279 | (719 << 16));
        graphics
            .context_registers
            .insert(super::AGC_DB_Z_READ_BASE, 0x0012_3456);
        super::agc_resolve_targetless_scanout(&mut graphics, &mut draw, true, Some(target));
        assert_eq!(
            draw.scanout_target, None,
            "a complete active depth target is not inferred as scanout"
        );

        graphics.context_registers.clear();
        super::agc_resolve_targetless_scanout(&mut graphics, &mut draw, true, None);
        assert_eq!(
            draw.scanout_target, None,
            "association never crosses a missing flip"
        );

        let pixel = draw.pixel_shader.as_mut().expect("test pixel shader");
        let mut instructions = pixel.program.instructions.clone();
        instructions.push(remu_shader::Instruction {
            pc: 8,
            encoding: remu_shader::ShaderEncoding::Exp,
            opcode: remu_shader::Opcode::EXPORT,
            words: vec![0, 0],
            sources: Vec::new(),
            destinations: Vec::new(),
            control: Some(remu_shader::InstructionControl::Export(
                remu_shader::ExportControl {
                    target: 1,
                    enable_mask: 0xf,
                    compressed: false,
                    done: true,
                    valid_mask: false,
                },
            )),
        });
        pixel.program = Arc::new(remu_shader::Program::new(
            pixel.program.address,
            instructions,
        ));
        draw.context_registers
            .iter_mut()
            .find(|register| register.offset == super::AGC_CB_SHADER_MASK)
            .expect("test shader mask")
            .value = 0xff;
        super::agc_resolve_targetless_scanout(&mut graphics, &mut draw, true, Some(target));
        assert_eq!(
            draw.scanout_target, None,
            "multiple color exports are ambiguous"
        );
    }

    #[test]
    fn resolves_constant_targetless_draw_before_ordered_barrier_acquire_and_flip() {
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
        let pixel_code = AlignedComputeProgram([0; 4]);
        let pixel_address = pixel_code.0.as_ptr().addr() as u64;
        let (mut graphics, mut shaders) =
            targetless_scanout_test_shader_state(pixel_address, target);
        let video_out = targetless_scanout_test_video(address);
        let (commands, expected_acquire) = targetless_barrier_acquire_flip_commands();
        let (sender, receiver) = remu_gpu::gpu_event_channel();
        remu_gpu::install_event_sink(sender);

        let mut submission_count = 0;
        let mut flips = Vec::new();
        let mut event_writes = Vec::new();
        super::agc_parse_submission(
            commands.as_ptr().addr() as u64,
            u32::try_from(commands.len()).expect("command dword count"),
            &mut super::AgcSubmissionContext {
                graphics: &mut graphics,
                shaders: &mut shaders,
                index_resources: &BTreeMap::new(),
                video_out: &video_out,
                submission_count: &mut submission_count,
                flips: &mut flips,
                event_writes: &mut event_writes,
            },
        )
        .expect("ordered targetless submission");
        remu_gpu::clear_event_sink();

        let remu_gpu::GpuEvent::IndexedDraw(draw) = receiver.recv().expect("resolved draw event")
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
        assert_eq!(
            receiver.recv().expect("pixel partial flush"),
            remu_gpu::GpuEvent::PipelineBarrier(remu_gpu::GpuPipelineBarrier {
                wait_for: remu_gpu::GpuPipelineStage::PixelShader,
            })
        );
        assert_eq!(
            receiver.recv().expect("memory acquire"),
            remu_gpu::GpuEvent::MemoryAcquire(expected_acquire)
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
        assert_eq!(flips, vec![(1, 0, 0x1234)]);
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
        let (sender, receiver) = remu_gpu::gpu_event_channel();
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
    fn builds_the_three_argument_gen5_index_buffer_packets() {
        let mut command_buffer = vec![0_u8; 0x40];
        let mut commands = vec![0_u32; 16];
        let indices = [0_u32, 1, 2, 1, 2, 3];
        let command_buffer_address =
            u64::try_from(command_buffer.as_mut_ptr().addr()).expect("command buffer address");
        let commands_address =
            u64::try_from(commands.as_mut_ptr().addr()).expect("command storage address");
        let index_buffer_address =
            u64::try_from(indices.as_ptr().addr()).expect("index buffer address");
        write_guest_u64(command_buffer_address + 0x10, commands_address);
        write_guest_u64(command_buffer_address + 0x18, commands_address + 64);
        write_guest_u32(command_buffer_address + 0x30, 0);

        let mut state = ProbeDispatchState::default();
        let command = agc_dcb_set_index_buffer(
            [command_buffer_address, index_buffer_address, 16, 0, 0, 0],
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
            Some(&index_buffer_address)
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
    use std::sync::Arc;

    use remu_core::GuestAddress;
    use remu_loader::load_reader;

    use super::{
        LinkedImage, host_protection_from_guest, process_mapping_contains,
        set_current_guest_thread_pointer,
    };

    type TlsHelper = unsafe extern "C" fn() -> u64;

    #[test]
    fn av_player_nv12_layout_matches_linear_gpu_texture_pitch() {
        assert_eq!(
            super::av_player_nv12_layout(1_440, 1_080),
            Some((1_536, 2_488_320))
        );
        assert_eq!(
            super::av_player_nv12_layout(1_920, 1_080),
            Some((2_048, 3_317_760))
        );
    }

    #[test]
    fn shader_evaluation_cache_buckets_vertex_record_extents_safely() {
        assert_eq!(super::canonical_vertex_record_count(None), None);
        assert_eq!(super::canonical_vertex_record_count(Some(0)), Some(1));
        assert_eq!(super::canonical_vertex_record_count(Some(1)), Some(1));
        assert_eq!(super::canonical_vertex_record_count(Some(513)), Some(1024));
        assert_eq!(super::canonical_vertex_record_count(Some(700)), Some(1024));
        assert_eq!(
            super::canonical_vertex_record_count(Some(u32::MAX)),
            Some(u32::MAX)
        );
    }

    #[test]
    fn shader_evaluation_cache_key_ignores_unreferenced_user_data() {
        let state = remu_shader::ShaderState {
            program: Arc::new(remu_shader::Program::new(0x1000, Vec::new())),
            user_data: vec![10, 20, 30, 40].into(),
            compute_system_registers: None,
            // Empty programs do not observe any incoming scalar register values.
            user_data_scalar_register_base: 104,
        };
        assert!(
            super::shader_evaluation_user_data_key(&state)
                .as_slice()
                .is_empty()
        );
    }

    #[test]
    fn shader_evaluation_cache_key_indexes_user_data_relative_to_scalar_base() {
        let program = remu_shader::Program::new(
            0x1000,
            vec![remu_shader::Instruction {
                pc: 0,
                encoding: remu_shader::ShaderEncoding::Smem,
                opcode: remu_shader::Opcode::END_PROGRAM,
                words: vec![0],
                sources: vec![remu_shader::Operand::scalar(104)],
                destinations: Vec::new(),
                control: Some(remu_shader::InstructionControl::ScalarMemory(
                    remu_shader::ScalarMemoryControl {
                        destination_count: 4,
                        immediate_offset_bytes: 0,
                        dynamic_offset_register: None,
                    },
                )),
            }],
        );
        let state = remu_shader::ShaderState {
            program: Arc::new(program),
            user_data: vec![10, 20, 30, 40].into(),
            compute_system_registers: None,
            user_data_scalar_register_base: 104,
        };

        assert_eq!(
            super::shader_evaluation_user_data_key(&state).as_slice(),
            [10, 20, 30, 40]
        );
    }

    #[test]
    fn recognizes_process_image_permission_changes_without_claiming_external_vmas() {
        assert!(process_mapping_contains(
            0x8000_0000,
            0x8002_0000,
            0x8000_4000,
            0x8000_8000
        ));
        assert!(!process_mapping_contains(
            0x8000_0000,
            0x8002_0000,
            0x7fff_c000,
            0x8000_4000
        ));
        assert!(!process_mapping_contains(
            0x8000_0000,
            0x8002_0000,
            0x8001_c000,
            0x8002_4000
        ));
        assert_eq!(
            host_protection_from_guest(0x12),
            libc::PROT_READ | libc::PROT_WRITE
        );
        assert_eq!(host_protection_from_guest(0x11), libc::PROT_READ);
    }

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
