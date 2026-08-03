// Copyright (C) 2026 REmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

use super::{
    MetalPresenter, PersistentRenderPipelineArchive, RenderPipelineArchiveState,
    ShaderFunctionCache, ShaderFunctionCacheEntry, ShaderFunctionStageCache,
    compile_metal_shader_library, invalidate_metal_library_cache, metal_cache_root,
};
use metal::{
    BinaryArchiveDescriptor, CompileOptions, ComputePipelineDescriptor, ComputePipelineState,
    Device, Function, RenderPipelineDescriptor, RenderPipelineState, URL,
};
use sha1::{Digest, Sha1};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, RwLock, TryLockError};
use std::time::{Duration, Instant};

impl MetalPresenter {
    pub(crate) fn persist_render_pipeline_archive(&self) {
        if let Some(archive) = self.render_pipeline_archive.as_ref()
            && let Err(error) = archive.serialize()
        {
            eprintln!("[REMu][METAL] {error}");
        }
    }
}

pub(crate) const METAL_RENDER_PIPELINE_ARCHIVE_FORMAT: &[u8] = b"remu-metal-render-pipeline-v1\0";

pub(crate) const SLOW_PIPELINE_ARCHIVE_ADMISSION: Duration = Duration::from_millis(100);

/// Persist enough cold-start progress to make a bounded follow-up run useful.
///
/// The runtime's diagnostic watchdog deliberately uses `_exit` as its final backstop, so it
/// cannot rely on presenter teardown to serialize a newly-created archive. Keep this high enough
/// to avoid rewriting a growing Unreal archive during the first splash shaders, but below the
/// number of variants a typical title compiles before its first scene transition.
pub(crate) const NEW_PIPELINE_ARCHIVE_CHECKPOINT_ADDITIONS: usize = 64;

pub(crate) const LOADED_PIPELINE_ARCHIVE_CHECKPOINT_ADDITIONS: usize = 2;

impl RenderPipelineArchiveState {
    fn record_addition_and_take_checkpoint(&mut self, threshold: usize) -> bool {
        self.additions = self.additions.saturating_add(1);
        if self.checkpoint_serialized || self.additions < threshold {
            return false;
        }
        self.checkpoint_serialized = true;
        true
    }

    fn quarantine_once(&mut self) -> bool {
        if self.quarantined {
            return false;
        }
        self.quarantined = true;
        true
    }
}

impl PersistentRenderPipelineArchive {
    pub(crate) fn open(device: &Device) -> Result<Self, String> {
        let path = metal_render_pipeline_archive_path(device);
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent).map_err(|error| {
                format!(
                    "could not create Metal pipeline cache {}: {error}",
                    parent.display()
                )
            })?;
        }
        // metal-rs 0.33 wraps +[NSURL URLWithString:] (an autoreleased object) as an owned URL.
        // Suppress the wrapper's release and let the active Objective-C pool perform the single
        // balanced release; otherwise archive serialization crashes at autoreleasePoolPop.
        let url = std::mem::ManuallyDrop::new(URL::new_with_string(&format!(
            "file://{}",
            path.display()
        )));
        let descriptor = BinaryArchiveDescriptor::new();
        let mut loaded = path.is_file();
        if loaded {
            descriptor.set_url(&url);
        }
        let archive = match device.new_binary_archive_with_descriptor(&descriptor) {
            Ok(archive) => archive,
            Err(error) if loaded => {
                eprintln!(
                    "[REMu][METAL] discarding unreadable pipeline archive {}: {error}",
                    path.display()
                );
                std::fs::remove_file(&path).map_err(|remove_error| {
                    format!(
                        "could not remove invalid pipeline archive {}: {remove_error}",
                        path.display()
                    )
                })?;
                loaded = false;
                device
                    .new_binary_archive_with_descriptor(&BinaryArchiveDescriptor::new())
                    .map_err(|error| format!("could not create Metal pipeline archive: {error}"))?
            }
            Err(error) => return Err(format!("could not create Metal pipeline archive: {error}")),
        };
        archive.set_label("REMu persistent render pipelines");
        eprintln!(
            "[REMu][METAL] {} render pipeline archive {}",
            if loaded { "loaded" } else { "created" },
            path.display()
        );
        eprintln!("[REMu][METAL] persistent compute pipelines share the render archive");
        Ok(Self {
            archive,
            path: Arc::new(path),
            loaded,
            state: Arc::new(Mutex::new(RenderPipelineArchiveState::default())),
            driver_lock: Arc::new(RwLock::new(())),
        })
    }

    pub(crate) fn new_render_pipeline_state(
        &self,
        device: &Device,
        make_descriptor: impl FnOnce() -> Result<RenderPipelineDescriptor, String>,
    ) -> Result<RenderPipelineState, String> {
        if self.loaded {
            // Cached pipeline creation remains concurrent. Only a demonstrably slow archive miss
            // is admitted after creation, under the exclusive lock, so an old archive can learn
            // shaders from later scenes without serializing every ordinary cache hit.
            let descriptor = make_descriptor()?;
            let started = Instant::now();
            let (pipeline, used_archive) = {
                let _reader = self
                    .driver_lock
                    .read()
                    .unwrap_or_else(std::sync::PoisonError::into_inner);
                let used_archive = !self.archive_is_quarantined();
                if used_archive {
                    descriptor.set_binary_archives(&[&self.archive]);
                }
                let pipeline = device
                    .new_render_pipeline_state(&descriptor)
                    .map_err(|error| format!("Metal render pipeline: {error}"))?;
                (pipeline, used_archive)
            };
            if used_archive && started.elapsed() >= SLOW_PIPELINE_ARCHIVE_ADMISSION {
                // Archive admission is cache maintenance, not a dependency of the pipeline that
                // was just created. Four scene-load compiler workers can all miss together; if
                // each waits for an exclusive archive mutation here, otherwise-ready pipelines
                // remain withheld from the presenter for over a second. Admit opportunistically
                // and let a later occurrence or clean shutdown persist skipped variants.
                let _writer = match self.driver_lock.try_write() {
                    Ok(writer) => writer,
                    Err(TryLockError::WouldBlock) => return Ok(pipeline),
                    Err(TryLockError::Poisoned(error)) => error.into_inner(),
                };
                if self.archive_is_quarantined() {
                    return Ok(pipeline);
                }
                match self
                    .archive
                    .add_render_pipeline_functions_with_descriptor(&descriptor)
                {
                    Ok(true) => {
                        let should_serialize = {
                            let mut state = self
                                .state
                                .lock()
                                .unwrap_or_else(std::sync::PoisonError::into_inner);
                            state.record_addition_and_take_checkpoint(
                                LOADED_PIPELINE_ARCHIVE_CHECKPOINT_ADDITIONS,
                            )
                        };
                        if should_serialize
                            && let Err(error) = self.serialize_or_quarantine_locked()
                        {
                            eprintln!(
                                "[REMu][METAL] slow-pipeline archive checkpoint failed: {error}"
                            );
                        }
                    }
                    Ok(false) => {}
                    Err(error) => eprintln!(
                        "[REMu][METAL] could not admit slow pipeline to loaded archive: {error}"
                    ),
                }
            }
            return Ok(pipeline);
        }
        // MTLBinaryArchive is documented as a resource shared by pipeline descriptors, but its
        // Rosetta driver path crashes while autoreleased descriptor internals from one thread are
        // drained concurrently with another archive mutation. Keep just descriptor/archive/
        // pipeline creation serialized and drain its local pool before admitting the next worker;
        // MSL lowering and metallib loading remain parallel outside this critical section.
        let _driver = self
            .driver_lock
            .write()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        objc::rc::autoreleasepool(|| {
            let descriptor = make_descriptor()?;
            descriptor.set_binary_archives(&[&self.archive]);
            self.archive
                .add_render_pipeline_functions_with_descriptor(&descriptor)
                .map_err(|error| {
                    format!("could not add Metal render pipeline to archive: {error}")
                })?;
            let should_serialize = {
                let mut state = self
                    .state
                    .lock()
                    .unwrap_or_else(std::sync::PoisonError::into_inner);
                state.record_addition_and_take_checkpoint(
                    NEW_PIPELINE_ARCHIVE_CHECKPOINT_ADDITIONS,
                )
            };
            let pipeline = device
                .new_render_pipeline_state(&descriptor)
                .map_err(|error| format!("Metal render pipeline: {error}"))?;
            // Diagnostic and interactive launches often end through SIGALRM or a window close
            // rather than a clean guest shutdown, so retain one mid-run checkpoint. The archive
            // can exceed 100 MiB for Unreal titles; rewriting it at every power of two additions
            // repeatedly stalls every compiler worker behind `driver_lock` during scene changes.
            // One checkpoint saves useful warm-up progress without turning a title transition
            // into nearly a gigabyte of synchronous archive writes. Clean shutdown still performs
            // the final serialization below.
            if should_serialize {
                self.serialize_or_quarantine_locked()?;
            }
            Ok(pipeline)
        })
    }

    pub(crate) fn new_compute_pipeline_state(
        &self,
        device: &Device,
        function: &Function,
    ) -> Result<ComputePipelineState, String> {
        let make_descriptor = || {
            let descriptor = ComputePipelineDescriptor::new();
            descriptor.set_label("REMu Gen5 compute");
            descriptor.set_compute_function(Some(function));
            descriptor
        };
        if self.loaded {
            // Match the render path: keep ordinary archive hits concurrent, then teach the loaded
            // archive only about a demonstrably slow miss. Compute scene transitions otherwise
            // pay the same pipeline specialization cost again even though their metallibs persist.
            let descriptor = make_descriptor();
            let started = Instant::now();
            let (pipeline, used_archive) = {
                let _reader = self
                    .driver_lock
                    .read()
                    .unwrap_or_else(std::sync::PoisonError::into_inner);
                let used_archive = !self.archive_is_quarantined();
                if used_archive {
                    descriptor.set_binary_archives(&[&self.archive]);
                }
                let pipeline = device
                    .new_compute_pipeline_state(&descriptor)
                    .map_err(|error| format!("Metal compute pipeline: {error}"))?;
                (pipeline, used_archive)
            };
            if used_archive && started.elapsed() >= SLOW_PIPELINE_ARCHIVE_ADMISSION {
                let _writer = match self.driver_lock.try_write() {
                    Ok(writer) => writer,
                    Err(TryLockError::WouldBlock) => return Ok(pipeline),
                    Err(TryLockError::Poisoned(error)) => error.into_inner(),
                };
                if self.archive_is_quarantined() {
                    return Ok(pipeline);
                }
                match self
                    .archive
                    .add_compute_pipeline_functions_with_descriptor(&descriptor)
                {
                    Ok(true) => {
                        let should_serialize = {
                            let mut state = self
                                .state
                                .lock()
                                .unwrap_or_else(std::sync::PoisonError::into_inner);
                            state.record_addition_and_take_checkpoint(
                                LOADED_PIPELINE_ARCHIVE_CHECKPOINT_ADDITIONS,
                            )
                        };
                        if should_serialize
                            && let Err(error) = self.serialize_or_quarantine_locked()
                        {
                            eprintln!(
                                "[REMu][METAL] slow-compute archive checkpoint failed: {error}"
                            );
                        }
                    }
                    Ok(false) => {}
                    Err(error) => eprintln!(
                        "[REMu][METAL] could not admit slow compute pipeline to loaded archive: {error}"
                    ),
                }
            }
            return Ok(pipeline);
        }

        // A fresh archive must learn the descriptor before pipeline creation. Share the render
        // driver's exclusive region so render and compute descriptors cannot race archive
        // mutation or autorelease cleanup on the Rosetta Metal path.
        let _driver = self
            .driver_lock
            .write()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        objc::rc::autoreleasepool(|| {
            let descriptor = make_descriptor();
            descriptor.set_binary_archives(&[&self.archive]);
            self.archive
                .add_compute_pipeline_functions_with_descriptor(&descriptor)
                .map_err(|error| {
                    format!("could not add Metal compute pipeline to archive: {error}")
                })?;
            let should_serialize = {
                let mut state = self
                    .state
                    .lock()
                    .unwrap_or_else(std::sync::PoisonError::into_inner);
                state.record_addition_and_take_checkpoint(NEW_PIPELINE_ARCHIVE_CHECKPOINT_ADDITIONS)
            };
            let pipeline = device
                .new_compute_pipeline_state(&descriptor)
                .map_err(|error| format!("Metal compute pipeline: {error}"))?;
            if should_serialize {
                self.serialize_or_quarantine_locked()?;
            }
            Ok(pipeline)
        })
    }

    fn archive_is_quarantined(&self) -> bool {
        self.state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .quarantined
    }

    fn serialize_locked(&self) -> Result<(), String> {
        let url = std::mem::ManuallyDrop::new(URL::new_with_string(&format!(
            "file://{}",
            self.path.display()
        )));
        self.archive
            .serialize_to_url(&url)
            .map(|_| ())
            .map_err(|error| {
                format!(
                    "could not persist Metal render pipeline archive {}: {error}",
                    self.path.display()
                )
            })
    }

    fn serialize_or_quarantine_locked(&self) -> Result<(), String> {
        if self.archive_is_quarantined() {
            return Ok(());
        }
        let error = match self.serialize_locked() {
            Ok(()) => return Ok(()),
            Err(error) => error,
        };
        // A newly-created archive failing to serialize points at the current descriptor/compiler
        // path and must remain a visible failure. Only a loaded on-disk archive can be stale or
        // structurally poisoned before this run.
        if !self.loaded || !archive_serialization_failure_indicates_poison(&error) {
            return Err(error);
        }
        {
            let mut state = self
                .state
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            if !state.quarantine_once() {
                return Err(error);
            }
        }
        match quarantine_pipeline_archive(&self.path) {
            Ok(Some(quarantine)) => eprintln!(
                "[REMu][METAL] quarantined invalid pipeline archive {} as {}; current-run pipelines will compile without it",
                self.path.display(),
                quarantine.display()
            ),
            Ok(None) => eprintln!(
                "[REMu][METAL] invalid pipeline archive {} was already absent; current-run pipelines will compile without it",
                self.path.display()
            ),
            Err(quarantine_error) => eprintln!(
                "[REMu][METAL] could not quarantine invalid pipeline archive {}: {quarantine_error}; current-run pipelines will compile without it",
                self.path.display()
            ),
        }
        Err(error)
    }

    pub(crate) fn serialize(&self) -> Result<(), String> {
        let state = self
            .state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        if state.additions == 0 || state.quarantined {
            return Ok(());
        }
        drop(state);
        let _driver = self
            .driver_lock
            .write()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        self.serialize_or_quarantine_locked()
    }
}

fn archive_serialization_failure_indicates_poison(error: &str) -> bool {
    let error = error.to_ascii_lowercase();
    (error.contains("expecting '") && error.contains(" stage in pipeline no."))
        || error.contains("binary archive is invalid")
        || error.contains("binary archive is corrupt")
        || error.contains("malformed binary archive")
        || error.contains("incompatible binary archive")
}

fn pipeline_archive_quarantine_path(path: &Path, ordinal: usize) -> PathBuf {
    let suffix = if ordinal == 0 {
        ".invalid".to_owned()
    } else {
        format!(".invalid-{ordinal}")
    };
    let mut name = path
        .file_name()
        .map_or_else(|| "pipeline-archive".into(), std::ffi::OsString::from);
    name.push(suffix);
    path.with_file_name(name)
}

fn quarantine_pipeline_archive(path: &Path) -> Result<Option<PathBuf>, String> {
    if !path.is_file() {
        return Ok(None);
    }
    for ordinal in 0..1_024 {
        let quarantine = pipeline_archive_quarantine_path(path, ordinal);
        match std::fs::hard_link(path, &quarantine) {
            Ok(()) => {
                std::fs::remove_file(path).map_err(|error| {
                    format!(
                        "linked {} but could not remove the active cache name: {error}",
                        quarantine.display()
                    )
                })?;
                return Ok(Some(quarantine));
            }
            Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => continue,
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(None),
            Err(error) => {
                return Err(format!(
                    "could not link {} to {}: {error}",
                    path.display(),
                    quarantine.display()
                ));
            }
        }
    }
    Err(format!(
        "all quarantine names for {} are already occupied",
        path.display()
    ))
}

impl ShaderFunctionCache {
    pub(crate) fn function(
        cache: &ShaderFunctionStageCache,
        device: &Device,
        stage: &str,
        shader: &remu_shader_metal::MetalShader,
    ) -> Result<(Function, bool), String> {
        let mut entries = cache
            .entries
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        loop {
            match entries.get(&shader.source) {
                Some(ShaderFunctionCacheEntry::Ready(function)) => {
                    return Ok((function.clone(), true));
                }
                Some(ShaderFunctionCacheEntry::Failed(error)) => return Err(error.clone()),
                Some(ShaderFunctionCacheEntry::Compiling) => {
                    entries = cache
                        .completed
                        .wait(entries)
                        .unwrap_or_else(std::sync::PoisonError::into_inner);
                }
                None => {
                    entries.insert(shader.source.clone(), ShaderFunctionCacheEntry::Compiling);
                    break;
                }
            }
        }

        // Pipeline variants often reach different compiler workers together while sharing an
        // identical stage. Claim the exact source above before leaving the lock, so only one
        // worker invokes Apple's compiler or loads the persistent metallib. Other workers wait
        // for that result instead of compiling the same 100-400 KiB source concurrently.
        drop(entries);
        let compiled = (|| {
            let library = compile_metal_shader_library(device, stage, &shader.source)?;
            library
                .get_function(&shader.entry_point, None)
                .map_err(|error| format!("Metal {stage} function: {error}"))
        })();
        let compiled = match compiled {
            Ok(function) => Ok(function),
            Err(error)
                if std::env::var_os("REMU_METAL_COMPILER").is_some()
                    && error.contains("does not exist") =>
            {
                invalidate_metal_library_cache(&shader.source);
                eprintln!(
                    "[REMu][METAL] discarding poisoned offline {stage} cache; falling back to in-process compile"
                );
                let compile_options = CompileOptions::new();
                compile_options.set_fast_math_enabled(false);
                let library = device
                    .new_library_with_source(&shader.source, &compile_options)
                    .map_err(|error| format!("Metal {stage} compilation: {error}"))?;
                library
                    .get_function(&shader.entry_point, None)
                    .map_err(|error| format!("Metal {stage} function: {error}"))
            }
            Err(error) => Err(error),
        };
        let mut entries = cache
            .entries
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        match &compiled {
            Ok(function) => {
                entries.insert(
                    shader.source.clone(),
                    ShaderFunctionCacheEntry::Ready(function.clone()),
                );
            }
            Err(error) => {
                entries.insert(
                    shader.source.clone(),
                    ShaderFunctionCacheEntry::Failed(error.clone()),
                );
            }
        }
        cache.completed.notify_all();
        compiled.map(|function| (function, false))
    }

    pub(crate) fn vertex(
        &self,
        device: &Device,
        shader: &remu_shader_metal::MetalShader,
    ) -> Result<(Function, bool), String> {
        Self::function(&self.vertex, device, "vertex", shader)
    }

    pub(crate) fn pixel(
        &self,
        device: &Device,
        shader: &remu_shader_metal::MetalShader,
    ) -> Result<(Function, bool), String> {
        Self::function(&self.pixel, device, "pixel", shader)
    }

    pub(crate) fn mesh(
        &self,
        device: &Device,
        shader: &remu_shader_metal::MetalShader,
    ) -> Result<(Function, bool), String> {
        Self::function(&self.mesh, device, "mesh", shader)
    }
}

pub(crate) fn metal_render_pipeline_archive_path(device: &Device) -> PathBuf {
    let mut hasher = Sha1::new();
    hasher.update(METAL_RENDER_PIPELINE_ARCHIVE_FORMAT);
    hasher.update(device.name().as_bytes());
    hasher.update(std::env::consts::ARCH.as_bytes());
    let digest = format!("{:x}", hasher.finalize());
    metal_cache_root().join(format!("render-{digest}.metallibarchive"))
}

#[cfg(test)]
mod tests {
    use super::{
        LOADED_PIPELINE_ARCHIVE_CHECKPOINT_ADDITIONS, NEW_PIPELINE_ARCHIVE_CHECKPOINT_ADDITIONS,
        PersistentRenderPipelineArchive, RenderPipelineArchiveState,
        archive_serialization_failure_indicates_poison, pipeline_archive_quarantine_path,
        quarantine_pipeline_archive,
    };
    use metal::{BinaryArchiveDescriptor, CompileOptions, Device};
    use std::path::PathBuf;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::{Arc, Mutex, RwLock};

    static NEXT_TEST_DIRECTORY: AtomicUsize = AtomicUsize::new(0);

    struct TestDirectory(PathBuf);

    impl TestDirectory {
        fn new() -> Self {
            let sequence = NEXT_TEST_DIRECTORY.fetch_add(1, Ordering::Relaxed);
            let path = std::env::temp_dir().join(format!(
                "remu-pipeline-archive-test-{}-{sequence}",
                std::process::id()
            ));
            std::fs::create_dir(&path).expect("create archive test directory");
            Self(path)
        }
    }

    impl Drop for TestDirectory {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    #[test]
    fn recognizes_structural_archive_poison_without_hiding_pipeline_or_io_failures() {
        assert!(archive_serialization_failure_indicates_poison(
            "Failed to generate machO: expecting 'fragment' stage in pipeline no. '42'"
        ));
        assert!(archive_serialization_failure_indicates_poison(
            "the binary archive is corrupt"
        ));
        assert!(!archive_serialization_failure_indicates_poison(
            "Metal render pipeline: fragment function failed to compile"
        ));
        assert!(!archive_serialization_failure_indicates_poison(
            "could not persist pipeline archive: permission denied"
        ));
    }

    #[test]
    fn archive_poison_is_quarantined_only_once_per_run() {
        let mut state = RenderPipelineArchiveState::default();
        assert!(state.quarantine_once());
        assert!(!state.quarantine_once());
        assert!(state.quarantined);
    }

    #[test]
    fn new_archive_checkpoints_once_after_bounded_cold_start_progress() {
        let mut state = RenderPipelineArchiveState::default();
        for _ in 1..NEW_PIPELINE_ARCHIVE_CHECKPOINT_ADDITIONS {
            assert!(
                !state
                    .record_addition_and_take_checkpoint(NEW_PIPELINE_ARCHIVE_CHECKPOINT_ADDITIONS)
            );
        }
        assert!(
            state.record_addition_and_take_checkpoint(NEW_PIPELINE_ARCHIVE_CHECKPOINT_ADDITIONS)
        );
        assert!(
            !state.record_addition_and_take_checkpoint(NEW_PIPELINE_ARCHIVE_CHECKPOINT_ADDITIONS)
        );
        assert_eq!(
            state.additions,
            NEW_PIPELINE_ARCHIVE_CHECKPOINT_ADDITIONS + 1
        );
    }

    #[test]
    fn loaded_archive_keeps_its_early_scene_variant_checkpoint() {
        let mut state = RenderPipelineArchiveState::default();
        assert!(
            !state
                .record_addition_and_take_checkpoint(LOADED_PIPELINE_ARCHIVE_CHECKPOINT_ADDITIONS)
        );
        assert!(
            state.record_addition_and_take_checkpoint(LOADED_PIPELINE_ARCHIVE_CHECKPOINT_ADDITIONS)
        );
        assert!(
            !state
                .record_addition_and_take_checkpoint(LOADED_PIPELINE_ARCHIVE_CHECKPOINT_ADDITIONS)
        );
    }

    #[test]
    fn fresh_archive_accepts_compute_pipeline_descriptors() {
        let Some(device) = Device::system_default() else {
            return;
        };
        objc::rc::autoreleasepool(|| {
            let archive = device
                .new_binary_archive_with_descriptor(&BinaryArchiveDescriptor::new())
                .expect("create test binary archive");
            let persistent = PersistentRenderPipelineArchive {
                archive,
                path: Arc::new(PathBuf::from("unused-compute-test.metallibarchive")),
                loaded: false,
                state: Arc::new(Mutex::new(RenderPipelineArchiveState::default())),
                driver_lock: Arc::new(RwLock::new(())),
            };
            let options = CompileOptions::new();
            let library = device
                .new_library_with_source(
                    concat!(
                        "#include <metal_stdlib>\n",
                        "using namespace metal;\n",
                        "kernel void remu_compute_test() {}\n",
                    ),
                    &options,
                )
                .expect("compile test compute library");
            let function = library
                .get_function("remu_compute_test", None)
                .expect("load test compute function");

            persistent
                .new_compute_pipeline_state(&device, &function)
                .expect("archive-backed compute pipeline");

            let state = persistent
                .state
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            assert_eq!(state.additions, 1);
        });
    }

    #[test]
    fn quarantines_without_overwriting_an_earlier_invalid_archive() {
        let directory = TestDirectory::new();
        let archive = directory.0.join("render.metallibarchive");
        let first_quarantine = pipeline_archive_quarantine_path(&archive, 0);
        std::fs::write(&archive, b"current invalid archive").expect("write active archive");
        std::fs::write(&first_quarantine, b"older invalid archive")
            .expect("write earlier quarantine");

        let quarantine = quarantine_pipeline_archive(&archive)
            .expect("quarantine invalid archive")
            .expect("active archive exists");

        assert_eq!(quarantine, pipeline_archive_quarantine_path(&archive, 1));
        assert!(!archive.exists());
        assert_eq!(
            std::fs::read(&quarantine).expect("read new quarantine"),
            b"current invalid archive"
        );
        assert_eq!(
            std::fs::read(&first_quarantine).expect("read earlier quarantine"),
            b"older invalid archive"
        );
    }

    #[test]
    fn absent_archive_is_already_quarantined() {
        let directory = TestDirectory::new();
        let archive = directory.0.join("missing.metallibarchive");
        assert_eq!(
            quarantine_pipeline_archive(&archive).expect("absent archive is harmless"),
            None
        );
    }
}
