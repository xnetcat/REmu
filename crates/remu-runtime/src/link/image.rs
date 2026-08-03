// Copyright (C) 2026 REmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

use crate::probe_entry;
use crate::tls::{TlsImageInput, install_tls_support, scan_tls_immediate_stores, static_tls_offsets};
use crate::unwind::{GuestUnwindModule, collect_unwind_fdes_at, guest_unwind_module};
use remu_core::GuestAddress;
use remu_loader::{LoadedExecutable, resolve_module_dependency_graph};
use remu_memory::GuestImage;
use std::collections::BTreeSet;
use std::path::Path;

use super::{
    ImportResolutionRecord, ImportResolutionStats, ImportStub, LinkError, LinkedGuestModule,
    ModuleLinkReport, PendingIfunc, RelocationStats, TlsRelocationContext, add_relocation_stats,
    append_object_initializers, apply_relocations_at, calculate_process_auxiliary_reserve,
    collect_import_resolutions, collect_process_exports, collect_tls_symbols, create_import_stubs,
    discover_runtime_module_paths, main_base_export_input, map_bundled_modules,
    module_exports_by_nid, resolve_process_param,
};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct GuestTls {
    pub thread_pointer: GuestAddress,
    pub static_block_size: u64,
    pub patched_loads: usize,
    pub patched_stores: usize,
}

#[derive(Debug)]
pub struct LinkedImage {
    pub memory: GuestImage,
    pub import_stubs: Vec<ImportStub>,
    pub relocation_stats: RelocationStats,
    pub tls: GuestTls,
    pub process_param: Option<GuestAddress>,
    pub(crate) import_resolution_stats: ImportResolutionStats,
    pub(crate) import_resolutions: Vec<ImportResolutionRecord>,
    pub(crate) unwind_fdes: Vec<GuestAddress>,
    #[cfg_attr(not(target_arch = "x86_64"), allow(dead_code))]
    pub(crate) module_initializers: Vec<GuestAddress>,
    #[cfg_attr(not(target_arch = "x86_64"), allow(dead_code))]
    pub(crate) bundled_modules: Vec<LinkedGuestModule>,
    #[cfg_attr(not(target_arch = "x86_64"), allow(dead_code))]
    pub(crate) unwind_modules: Vec<GuestUnwindModule>,
    /// Ifunc relocations whose guest resolvers run once at process entry (F2).
    #[cfg_attr(not(target_arch = "x86_64"), allow(dead_code))]
    pub(crate) pending_ifunc_relocations: Vec<PendingIfunc>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ProbeOutcome {
    pub return_value: i32,
    pub import_hits: usize,
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

    pub(super) fn link_main_with_modules_and_startup(
        executable: &LoadedExecutable,
        modules: &[(&str, &LoadedExecutable)],
        module_startup: &[bool],
    ) -> Result<Self, LinkError> {
        debug_assert_eq!(modules.len(), module_startup.len());
        let elf = executable.info.elf.as_ref().ok_or(LinkError::MissingElf)?;
        let dynamic = elf
            .dynamic
            .as_ref()
            .ok_or(LinkError::MissingDynamicMetadata)?;
        let tls_store_plan = scan_tls_immediate_stores(
            std::iter::once(executable).chain(modules.iter().map(|(_, executable)| *executable)),
        )?;
        let auxiliary_reserve = calculate_process_auxiliary_reserve(
            dynamic,
            modules,
            tls_store_plan.helper_region_size()?,
        )?;
        let mut memory =
            GuestImage::map_main_with_auxiliary_reserve(executable, auxiliary_reserve)?;
        let linked_modules = map_bundled_modules(&mut memory, modules)?;

        let main_export = main_base_export_input(executable, dynamic, memory.image_base());
        let process_exports = collect_process_exports(&memory, &main_export, &linked_modules)?;
        let (import_resolution_stats, import_resolutions) = collect_import_resolutions(
            std::iter::once(dynamic).chain(linked_modules.iter().map(|module| module.dynamic)),
            &process_exports,
        )?;
        // Lay out the static-TLS blocks up front so a TLS relocation can name the thread-pointer-
        // relative and defining-module offsets it needs before `install_tls_support` writes those
        // blocks. The image list is the same order the installer uses (main first, then modules).
        let main_base = memory.image_base();
        let tls_layout_inputs = std::iter::once(TlsImageInput {
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
        let module_static_offsets = static_tls_offsets(&tls_layout_inputs)?;
        let static_offset_of =
            |module_id: u64| module_static_offsets.get(&module_id).copied().unwrap_or(0);
        let tls_symbols = collect_tls_symbols(
            &std::iter::once((1_u64, static_offset_of(1), dynamic))
                .chain(linked_modules.iter().map(|module| {
                    (
                        module.tls_module_id,
                        static_offset_of(module.tls_module_id),
                        module.dynamic,
                    )
                }))
                .collect::<Vec<_>>(),
        )?;

        let mut import_stubs = Vec::new();
        let mut relocation_stats = RelocationStats::default();
        let mut pending_ifunc_relocations = Vec::new();
        let mut unwind_fdes = Vec::new();
        for module in &linked_modules {
            let (module_stubs, symbol_stubs) = create_import_stubs(
                &mut memory,
                module.dynamic,
                import_stubs.len(),
                &process_exports,
            )?;
            import_stubs.extend(module_stubs);
            let tls_context = TlsRelocationContext {
                module_id: module.tls_module_id,
                module_static_offset: static_offset_of(module.tls_module_id),
                tls_symbols: &tls_symbols,
            };
            add_relocation_stats(
                &mut relocation_stats,
                apply_relocations_at(
                    &mut memory,
                    module.base,
                    &tls_context,
                    module.dynamic,
                    &symbol_stubs,
                    &process_exports,
                    &mut pending_ifunc_relocations,
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
        let process_param = resolve_process_param(elf, main_base)?;
        let main_tls_context = TlsRelocationContext {
            module_id: 1,
            module_static_offset: static_offset_of(1),
            tls_symbols: &tls_symbols,
        };
        add_relocation_stats(
            &mut relocation_stats,
            apply_relocations_at(
                &mut memory,
                main_base,
                &main_tls_context,
                dynamic,
                &symbol_stubs,
                &process_exports,
                &mut pending_ifunc_relocations,
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
        let tls = install_tls_support(&mut memory, &tls_layout_inputs, &tls_store_plan)?;
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
            pending_ifunc_relocations,
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
