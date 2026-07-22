// SPDX-License-Identifier: GPL-2.0-or-later

//! Filesystem discovery and dependency ordering for bundled guest modules.

use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::io;
use std::path::{Component, Path, PathBuf};

use thiserror::Error;

use crate::{InspectError, SceLibraryReference, SceModuleReference, inspect_path};

/// The dependency metadata needed by the runtime for one guest executable or PRX.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ModuleDependencyNode {
    pub path: PathBuf,
    pub needed_libraries: Vec<String>,
    pub original_filename: Option<String>,
    pub required_modules: Vec<SceModuleReference>,
    pub exported_modules: Vec<SceModuleReference>,
    pub imported_libraries: Vec<SceLibraryReference>,
    pub exported_libraries: Vec<SceLibraryReference>,
}

/// A dependency for which no bundled PRX or SPRX exists in the active app view.
///
/// These entries are commonly system modules supplied by HLE. Keeping them explicit lets the
/// runtime distinguish a supported HLE dependency from a genuinely missing game module.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct UnresolvedModuleDependency {
    pub requested_by: PathBuf,
    pub name: String,
    pub searched_roots: Vec<PathBuf>,
}

/// A recursively resolved guest-module dependency graph.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ModuleDependencyGraph {
    pub app_root: PathBuf,
    pub search_roots: Vec<PathBuf>,
    pub entry: ModuleDependencyNode,
    /// Bundled modules in dependency-first order. The entry executable is not repeated here.
    pub dependency_order: Vec<ModuleDependencyNode>,
    pub unresolved_dependencies: Vec<UnresolvedModuleDependency>,
}

impl ModuleDependencyGraph {
    #[must_use]
    pub fn is_complete(&self) -> bool {
        self.unresolved_dependencies.is_empty()
    }
}

/// Configures module-graph discovery for an entry executable.
#[derive(Clone, Debug)]
pub struct ModuleDependencyGraphBuilder {
    entry_path: PathBuf,
    app_root: Option<PathBuf>,
    deny_unresolved: bool,
}

impl ModuleDependencyGraphBuilder {
    #[must_use]
    pub fn new(entry_path: impl AsRef<Path>) -> Self {
        Self {
            entry_path: entry_path.as_ref().to_path_buf(),
            app_root: None,
            deny_unresolved: false,
        }
    }

    /// Overrides automatic app-root discovery.
    #[must_use]
    pub fn app_root(mut self, app_root: impl AsRef<Path>) -> Self {
        self.app_root = Some(app_root.as_ref().to_path_buf());
        self
    }

    /// Makes an absent bundled dependency an error instead of recording it for HLE resolution.
    #[must_use]
    pub const fn deny_unresolved(mut self, deny: bool) -> Self {
        self.deny_unresolved = deny;
        self
    }

    pub fn resolve(self) -> Result<ModuleDependencyGraph, ModuleDependencyGraphError> {
        let entry_path = canonicalize(&self.entry_path)?;
        if !entry_path.is_file() {
            return Err(ModuleDependencyGraphError::EntryNotFile { path: entry_path });
        }
        let app_root = match self.app_root {
            Some(root) => canonicalize(&root)?,
            None => discover_app_root(&entry_path)?,
        };
        if !app_root.is_dir() {
            return Err(ModuleDependencyGraphError::AppRootNotDirectory { path: app_root });
        }
        if !entry_path.starts_with(&app_root) {
            return Err(ModuleDependencyGraphError::EntryOutsideAppRoot {
                entry: entry_path,
                app_root,
            });
        }

        let search_roots = module_search_roots(&entry_path, &app_root);
        let candidates = CandidateIndex::build(&app_root, &search_roots)?;
        let mut resolver = GraphResolver::new(candidates, self.deny_unresolved);
        resolver.visit(entry_path.clone())?;
        let entry_index = resolver
            .dependency_order
            .iter()
            .position(|node| node.path == entry_path)
            .ok_or_else(|| ModuleDependencyGraphError::InternalEntryMissing {
                entry: entry_path.clone(),
            })?;
        let entry = resolver.dependency_order.remove(entry_index);
        Ok(ModuleDependencyGraph {
            app_root,
            search_roots,
            entry,
            dependency_order: resolver.dependency_order,
            unresolved_dependencies: resolver.unresolved_dependencies,
        })
    }
}

/// Resolves all bundled dependencies of an executable and records absent dependencies for HLE.
pub fn resolve_module_dependency_graph(
    entry_path: impl AsRef<Path>,
) -> Result<ModuleDependencyGraph, ModuleDependencyGraphError> {
    ModuleDependencyGraphBuilder::new(entry_path).resolve()
}

/// Finds the app0 root containing an executable, including executables in decrypted overlays.
pub fn discover_app_root(
    entry_path: impl AsRef<Path>,
) -> Result<PathBuf, ModuleDependencyGraphError> {
    let entry = canonicalize(entry_path.as_ref())?;
    let Some(parent) = entry.parent() else {
        return Err(ModuleDependencyGraphError::AppRootNotFound {
            entry,
            checked: Vec::new(),
        });
    };
    let ancestors = parent
        .ancestors()
        .map(Path::to_path_buf)
        .collect::<Vec<_>>();

    // A decrypted overlay may contain a copied param file. Prefer the parent app0 marker in that
    // case instead of treating the overlay as an independent application.
    for directory in &ancestors {
        if path_file_name_eq(directory, "decrypted")
            && let Some(app_root) = directory.parent()
            && (app_root.join("sce_sys/param.json").is_file()
                || app_root.join("sce_sys/param.sfo").is_file())
        {
            return Ok(app_root.to_path_buf());
        }
    }

    // param.json and param.sfo belong to app0/sce_sys, making this stronger than directories that
    // are often copied into a decrypted overlay.
    for directory in &ancestors {
        if directory.join("sce_sys/param.json").is_file()
            || directory.join("sce_sys/param.sfo").is_file()
        {
            return Ok(directory.clone());
        }
    }

    // Dump tools conventionally put decrypted executables in app0/decrypted while leaving the
    // authoritative content root one level above.
    for directory in &ancestors {
        if path_file_name_eq(directory, "decrypted")
            && let Some(app_root) = directory.parent()
        {
            return Ok(app_root.to_path_buf());
        }
    }

    // When package metadata was omitted from a dump, choose the outermost app-like ancestor. This
    // avoids mistaking a copied sce_sys inside an overlay for the actual content root.
    if let Some(directory) = ancestors
        .iter()
        .rev()
        .find(|directory| directory.join("sce_sys").is_dir())
    {
        return Ok(directory.clone());
    }
    if path_file_name_eq(&entry, "eboot.bin") {
        return Ok(parent.to_path_buf());
    }

    Err(ModuleDependencyGraphError::AppRootNotFound {
        entry,
        checked: ancestors,
    })
}

#[derive(Debug, Error)]
pub enum ModuleDependencyGraphError {
    #[error("could not access {path}: {source}")]
    FileSystem {
        path: PathBuf,
        #[source]
        source: io::Error,
    },
    #[error("module-graph entry is not a file: {path}")]
    EntryNotFile { path: PathBuf },
    #[error("module-graph app root is not a directory: {path}")]
    AppRootNotDirectory { path: PathBuf },
    #[error("could not find an app root above {entry}; checked {checked:?}")]
    AppRootNotFound {
        entry: PathBuf,
        checked: Vec<PathBuf>,
    },
    #[error("entry {entry} is outside configured app root {app_root}")]
    EntryOutsideAppRoot { entry: PathBuf, app_root: PathBuf },
    #[error("module candidate {candidate} resolves outside app root {app_root}")]
    CandidateOutsideAppRoot {
        candidate: PathBuf,
        app_root: PathBuf,
    },
    #[error("module candidate has a non-Unicode file name: {path}")]
    NonUnicodeCandidate { path: PathBuf },
    #[error("could not inspect dependency metadata in {path}: {source}")]
    InspectModule {
        path: PathBuf,
        #[source]
        source: InspectError,
    },
    #[error("module candidate is not a PS5 ELF or executable container: {path}")]
    MissingElf { path: PathBuf },
    #[error("invalid dependency path {dependency:?} requested by {requested_by}")]
    InvalidDependencyPath {
        requested_by: PathBuf,
        dependency: String,
    },
    #[error("{requested_by} declares dependency {dependency:?} more than once")]
    DuplicateDependency {
        requested_by: PathBuf,
        dependency: String,
    },
    #[error(
        "dependency {dependency:?} requested by {requested_by} matches multiple bundled modules: {candidates:?}"
    )]
    DuplicateCandidates {
        requested_by: PathBuf,
        dependency: String,
        candidates: Vec<PathBuf>,
    },
    #[error(
        "dependency {dependency:?} requested by {requested_by} was not found under {searched_roots:?}"
    )]
    MissingDependency {
        requested_by: PathBuf,
        dependency: String,
        searched_roots: Vec<PathBuf>,
    },
    #[error("guest module dependency cycle detected: {cycle:?}")]
    DependencyCycle { cycle: Vec<PathBuf> },
    #[error("guest module identity {identity:?} is exported by both {first} and {second}")]
    DuplicateModuleIdentity {
        identity: String,
        first: PathBuf,
        second: PathBuf,
    },
    #[error("internal module-graph error: resolved entry {entry} is absent from its order")]
    InternalEntryMissing { entry: PathBuf },
}

fn canonicalize(path: &Path) -> Result<PathBuf, ModuleDependencyGraphError> {
    fs::canonicalize(path).map_err(|source| ModuleDependencyGraphError::FileSystem {
        path: path.to_path_buf(),
        source,
    })
}

fn module_search_roots(entry_path: &Path, app_root: &Path) -> Vec<PathBuf> {
    let entry_directory = entry_path.parent().unwrap_or(app_root);
    let overlay = entry_directory
        .ancestors()
        .take_while(|path| *path != app_root)
        .find(|directory| {
            directory.join("sce_module").is_dir()
                || directory.join("sce_modules").is_dir()
                || directory.join("Media/Modules").is_dir()
        });
    let mut roots = Vec::with_capacity(2);
    if let Some(overlay) = overlay {
        roots.push(overlay.to_path_buf());
    }
    roots.push(app_root.to_path_buf());
    roots
}

#[derive(Clone, Debug)]
struct ModuleCandidate {
    path: PathBuf,
    tier: usize,
}

#[derive(Debug)]
struct CandidateIndex {
    search_roots: Vec<PathBuf>,
    by_relative_path: BTreeMap<String, Vec<ModuleCandidate>>,
    by_file_name: BTreeMap<String, Vec<ModuleCandidate>>,
    by_file_stem: BTreeMap<String, Vec<ModuleCandidate>>,
}

impl CandidateIndex {
    fn build(
        app_root: &Path,
        search_roots: &[PathBuf],
    ) -> Result<Self, ModuleDependencyGraphError> {
        let mut index = Self {
            search_roots: search_roots.to_vec(),
            by_relative_path: BTreeMap::new(),
            by_file_name: BTreeMap::new(),
            by_file_stem: BTreeMap::new(),
        };
        let overlay = search_roots
            .first()
            .filter(|root| root.as_path() != app_root)
            .map(PathBuf::as_path);
        for (tier, root) in search_roots.iter().enumerate() {
            let excluded = if root == app_root { overlay } else { None };
            index.scan_root(app_root, root, tier, excluded)?;
        }
        Ok(index)
    }

    fn scan_root(
        &mut self,
        app_root: &Path,
        search_root: &Path,
        tier: usize,
        excluded: Option<&Path>,
    ) -> Result<(), ModuleDependencyGraphError> {
        let mut pending = vec![search_root.to_path_buf()];
        while let Some(directory) = pending.pop() {
            let entries = fs::read_dir(&directory)
                .map_err(|source| ModuleDependencyGraphError::FileSystem {
                    path: directory.clone(),
                    source,
                })?
                .collect::<Result<Vec<_>, _>>()
                .map_err(|source| ModuleDependencyGraphError::FileSystem {
                    path: directory.clone(),
                    source,
                })?;
            let mut entries = entries;
            entries.sort_by_key(fs::DirEntry::file_name);
            for entry in entries.into_iter().rev() {
                let path = entry.path();
                if excluded.is_some_and(|excluded| path == excluded) {
                    continue;
                }
                let file_type =
                    entry
                        .file_type()
                        .map_err(|source| ModuleDependencyGraphError::FileSystem {
                            path: path.clone(),
                            source,
                        })?;
                if file_type.is_dir() {
                    if !is_non_guest_directory(&path) {
                        pending.push(path);
                    }
                    continue;
                }
                if !file_type.is_file() && !file_type.is_symlink() {
                    continue;
                }
                if !is_module_path(&path) {
                    continue;
                }
                let canonical_path = canonicalize(&path)?;
                if !canonical_path.starts_with(app_root) {
                    return Err(ModuleDependencyGraphError::CandidateOutsideAppRoot {
                        candidate: canonical_path,
                        app_root: app_root.to_path_buf(),
                    });
                }
                let relative = path.strip_prefix(search_root).map_err(|_| {
                    ModuleDependencyGraphError::CandidateOutsideAppRoot {
                        candidate: path.clone(),
                        app_root: app_root.to_path_buf(),
                    }
                })?;
                let relative_path = path_key(relative).ok_or_else(|| {
                    ModuleDependencyGraphError::NonUnicodeCandidate { path: path.clone() }
                })?;
                let file_name = path
                    .file_name()
                    .and_then(|name| name.to_str())
                    .ok_or_else(|| ModuleDependencyGraphError::NonUnicodeCandidate {
                        path: path.clone(),
                    })?
                    .to_ascii_lowercase();
                let file_stem = path
                    .file_stem()
                    .and_then(|name| name.to_str())
                    .ok_or_else(|| ModuleDependencyGraphError::NonUnicodeCandidate {
                        path: path.clone(),
                    })?
                    .to_ascii_lowercase();
                let candidate = ModuleCandidate {
                    path: canonical_path,
                    tier,
                };
                self.by_relative_path
                    .entry(relative_path)
                    .or_default()
                    .push(candidate.clone());
                self.by_file_name
                    .entry(file_name)
                    .or_default()
                    .push(candidate.clone());
                self.by_file_stem
                    .entry(file_stem)
                    .or_default()
                    .push(candidate);
            }
        }
        Ok(())
    }

    fn resolve(
        &self,
        requested_by: &Path,
        dependency: &str,
    ) -> Result<Option<PathBuf>, ModuleDependencyGraphError> {
        let logical_path = dependency_path_key(dependency).ok_or_else(|| {
            ModuleDependencyGraphError::InvalidDependencyPath {
                requested_by: requested_by.to_path_buf(),
                dependency: dependency.to_owned(),
            }
        })?;
        let file_name = Path::new(&logical_path)
            .file_name()
            .and_then(|name| name.to_str())
            .ok_or_else(|| ModuleDependencyGraphError::InvalidDependencyPath {
                requested_by: requested_by.to_path_buf(),
                dependency: dependency.to_owned(),
            })?;
        let candidates = if logical_path.contains('/') {
            self.by_relative_path.get(&logical_path)
        } else {
            self.by_file_name.get(file_name).or_else(|| {
                let stem = Path::new(file_name).file_stem()?.to_str()?;
                self.by_file_stem.get(stem)
            })
        };
        let Some(candidates) = candidates else {
            return Ok(None);
        };
        let minimum_tier = candidates.iter().map(|candidate| candidate.tier).min();
        let mut paths = candidates
            .iter()
            .filter(|candidate| Some(candidate.tier) == minimum_tier)
            .map(|candidate| candidate.path.clone())
            .collect::<BTreeSet<_>>()
            .into_iter()
            .collect::<Vec<_>>();
        paths.sort();
        match paths.as_slice() {
            [] => Ok(None),
            [path] => Ok(Some(path.clone())),
            _ => Err(ModuleDependencyGraphError::DuplicateCandidates {
                requested_by: requested_by.to_path_buf(),
                dependency: dependency.to_owned(),
                candidates: paths,
            }),
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum VisitState {
    Visiting,
    Visited,
}

struct GraphResolver {
    candidates: CandidateIndex,
    deny_unresolved: bool,
    states: BTreeMap<PathBuf, VisitState>,
    stack: Vec<PathBuf>,
    module_identities: BTreeMap<String, PathBuf>,
    dependency_order: Vec<ModuleDependencyNode>,
    unresolved_dependencies: Vec<UnresolvedModuleDependency>,
}

impl GraphResolver {
    fn new(candidates: CandidateIndex, deny_unresolved: bool) -> Self {
        Self {
            candidates,
            deny_unresolved,
            states: BTreeMap::new(),
            stack: Vec::new(),
            module_identities: BTreeMap::new(),
            dependency_order: Vec::new(),
            unresolved_dependencies: Vec::new(),
        }
    }

    fn visit(&mut self, path: PathBuf) -> Result<(), ModuleDependencyGraphError> {
        match self.states.get(&path) {
            Some(VisitState::Visited) => return Ok(()),
            Some(VisitState::Visiting) => {
                let start = self
                    .stack
                    .iter()
                    .position(|entry| entry == &path)
                    .unwrap_or(0);
                let mut cycle = self.stack[start..].to_vec();
                cycle.push(path);
                return Err(ModuleDependencyGraphError::DependencyCycle { cycle });
            }
            None => {}
        }
        self.states.insert(path.clone(), VisitState::Visiting);
        self.stack.push(path.clone());

        let node = inspect_dependency_node(&path)?;
        self.register_module_identities(&node)?;
        let dependencies = unique_sorted_dependencies(&node)?;
        for dependency in dependencies {
            if let Some(candidate) = self.candidates.resolve(&path, &dependency)? {
                self.visit(candidate)?;
            } else if self.deny_unresolved {
                return Err(ModuleDependencyGraphError::MissingDependency {
                    requested_by: path.clone(),
                    dependency,
                    searched_roots: self.candidates.search_roots.clone(),
                });
            } else {
                self.unresolved_dependencies
                    .push(UnresolvedModuleDependency {
                        requested_by: path.clone(),
                        name: dependency,
                        searched_roots: self.candidates.search_roots.clone(),
                    });
            }
        }

        self.stack.pop();
        self.states.insert(path, VisitState::Visited);
        self.dependency_order.push(node);
        Ok(())
    }

    fn register_module_identities(
        &mut self,
        node: &ModuleDependencyNode,
    ) -> Result<(), ModuleDependencyGraphError> {
        for identity in &node.exported_modules {
            let key = identity.name.to_ascii_lowercase();
            if let Some(first) = self.module_identities.get(&key) {
                if first != &node.path {
                    return Err(ModuleDependencyGraphError::DuplicateModuleIdentity {
                        identity: identity.name.clone(),
                        first: first.clone(),
                        second: node.path.clone(),
                    });
                }
            } else {
                self.module_identities.insert(key, node.path.clone());
            }
        }
        Ok(())
    }
}

fn inspect_dependency_node(
    path: &Path,
) -> Result<ModuleDependencyNode, ModuleDependencyGraphError> {
    let info = inspect_path(path).map_err(|source| ModuleDependencyGraphError::InspectModule {
        path: path.to_path_buf(),
        source,
    })?;
    let elf = info
        .elf
        .ok_or_else(|| ModuleDependencyGraphError::MissingElf {
            path: path.to_path_buf(),
        })?;
    let Some(dynamic) = elf.dynamic else {
        return Ok(ModuleDependencyNode {
            path: path.to_path_buf(),
            needed_libraries: Vec::new(),
            original_filename: None,
            required_modules: Vec::new(),
            exported_modules: Vec::new(),
            imported_libraries: Vec::new(),
            exported_libraries: Vec::new(),
        });
    };
    Ok(ModuleDependencyNode {
        path: path.to_path_buf(),
        needed_libraries: dynamic.needed_libraries,
        original_filename: dynamic.original_filename,
        required_modules: dynamic.required_modules,
        exported_modules: dynamic.exported_modules,
        imported_libraries: dynamic.imported_libraries,
        exported_libraries: dynamic.exported_libraries,
    })
}

fn unique_sorted_dependencies(
    node: &ModuleDependencyNode,
) -> Result<Vec<String>, ModuleDependencyGraphError> {
    let mut dependencies = BTreeMap::new();
    for dependency in &node.needed_libraries {
        let key = dependency_path_key(dependency).ok_or_else(|| {
            ModuleDependencyGraphError::InvalidDependencyPath {
                requested_by: node.path.clone(),
                dependency: dependency.clone(),
            }
        })?;
        if dependencies.insert(key, dependency.clone()).is_some() {
            return Err(ModuleDependencyGraphError::DuplicateDependency {
                requested_by: node.path.clone(),
                dependency: dependency.clone(),
            });
        }
    }
    Ok(dependencies.into_values().collect())
}

fn dependency_path_key(dependency: &str) -> Option<String> {
    let normalized = dependency.replace('\\', "/");
    let lower = normalized.to_ascii_lowercase();
    let relative = if lower.starts_with("/app0/") {
        &normalized[6..]
    } else if lower.starts_with("app0/") {
        &normalized[5..]
    } else if normalized.starts_with('/') {
        return None;
    } else {
        &normalized
    };
    if relative.is_empty() {
        return None;
    }
    let mut components = Vec::new();
    for component in Path::new(relative).components() {
        match component {
            Component::Normal(value) => components.push(value.to_str()?.to_ascii_lowercase()),
            Component::CurDir => {}
            Component::ParentDir | Component::RootDir | Component::Prefix(_) => return None,
        }
    }
    if components.is_empty() {
        None
    } else {
        Some(components.join("/"))
    }
}

fn path_key(path: &Path) -> Option<String> {
    let mut components = Vec::new();
    for component in path.components() {
        if let Component::Normal(value) = component {
            components.push(value.to_str()?.to_ascii_lowercase());
        }
    }
    Some(components.join("/"))
}

fn is_module_path(path: &Path) -> bool {
    path.extension()
        .and_then(|extension| extension.to_str())
        .is_some_and(|extension| {
            extension.eq_ignore_ascii_case("prx") || extension.eq_ignore_ascii_case("sprx")
        })
}

fn is_non_guest_directory(path: &Path) -> bool {
    [".git", "target", "fakelib", "stubs"]
        .iter()
        .any(|name| path_file_name_eq(path, name))
}

fn path_file_name_eq(path: &Path, expected: &str) -> bool {
    path.file_name()
        .and_then(|name| name.to_str())
        .is_some_and(|name| name.eq_ignore_ascii_case(expected))
}
