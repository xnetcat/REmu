// SPDX-License-Identifier: GPL-2.0-or-later

//! Subcommands that materialize guest memory and run guest code.

use std::path::Path;
use std::process::ExitCode;
#[cfg(any(target_os = "macos", windows))]
use std::time::{Duration, Instant};

pub(crate) fn map(path: &Path) -> ExitCode {
    let executable = match remu_loader::load_path(path) {
        Ok(executable) => executable,
        Err(error) => {
            eprintln!("error: {}: {error}", path.display());
            return ExitCode::FAILURE;
        }
    };
    let mut image = match remu_memory::GuestImage::map_main(&executable) {
        Ok(image) => image,
        Err(error) => {
            eprintln!("error: {}: {error}", path.display());
            return ExitCode::FAILURE;
        }
    };
    println!("path={}", path.display());
    println!("image_base={}", image.image_base());
    println!("mapping_start={}", image.mapping_start());
    println!("mapping_size=0x{:x}", image.mapping_size());
    println!("entry={}", image.entry_address());
    for (index, region) in image.regions().iter().enumerate() {
        println!(
            "region[{index}]=start:{} file:0x{:x} memory:0x{:x} flags:0x{:x}",
            region.start, region.file_size, region.memory_size, region.flags
        );
    }
    if let Err(error) = image.finalize_permissions() {
        eprintln!("error: {}: {error}", path.display());
        return ExitCode::FAILURE;
    }
    println!("permissions=finalized");
    ExitCode::SUCCESS
}

pub(crate) fn link(path: &Path) -> ExitCode {
    let (linked, module_report) = match load_process_image(path) {
        Ok(process) => process,
        Err(error) => {
            eprintln!("error: {error}");
            return ExitCode::FAILURE;
        }
    };
    println!("path={}", path.display());
    println!("image_base={}", linked.memory.image_base());
    println!("entry={}", linked.memory.entry_address());
    println!("import_stubs={}", linked.import_stubs.len());
    let import_resolutions = linked.import_resolution_stats();
    println!("imports.bundled_guest={}", import_resolutions.bundled_guest);
    println!("imports.host_direct={}", import_resolutions.host_direct);
    println!("imports.hle_dispatch={}", import_resolutions.hle_dispatch);
    println!(
        "imports.unresolved_weak={}",
        import_resolutions.unresolved_weak
    );
    for resolution in linked.import_resolutions() {
        println!(
            "import_resolution={} nid={} weak={} symbol={}",
            resolution.kind.as_str(),
            resolution.nid,
            resolution.weak,
            resolution.symbol
        );
    }
    println!("unwind_records={}", linked.unwind_record_count());
    println!("bundled_modules={}", module_report.loaded_modules.len());
    println!(
        "system_module_dependencies={}",
        module_report.unresolved_dependencies.len()
    );
    println!("relocations={}", linked.relocation_stats.total());
    println!("tls.thread_pointer={}", linked.tls.thread_pointer);
    println!("tls.static_block_size=0x{:x}", linked.tls.static_block_size);
    println!("tls.patched_loads={}", linked.tls.patched_loads);
    println!("tls.patched_stores={}", linked.tls.patched_stores);
    println!("relocations.relative={}", linked.relocation_stats.relative);
    println!("relocations.absolute={}", linked.relocation_stats.absolute);
    println!(
        "relocations.global_data={}",
        linked.relocation_stats.global_data
    );
    println!(
        "relocations.jump_slots={}",
        linked.relocation_stats.jump_slots
    );
    println!("permissions=finalized");
    ExitCode::SUCCESS
}

pub(crate) fn probe(path: &Path, import_limit: usize, timeout_seconds: u32) -> ExitCode {
    let (linked, module_report) = match load_process_image(path) {
        Ok(process) => process,
        Err(error) => {
            eprintln!("error: {error}");
            return ExitCode::FAILURE;
        }
    };
    eprintln!(
        "[REMu][PROBE] entering {} at {} with {} import stubs, TLS {} ({} patched loads, {} patched stores)",
        path.display(),
        linked.memory.entry_address(),
        linked.import_stubs.len(),
        linked.tls.thread_pointer,
        linked.tls.patched_loads,
        linked.tls.patched_stores,
    );
    let image_name = module_report
        .entry_path
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or("eboot.bin");
    let outcome = linked.probe_entry(
        image_name,
        &module_report.app_root,
        import_limit,
        timeout_seconds,
    );
    remu_runtime::report_shader_opcode_coverage();
    match outcome {
        Ok(outcome) => {
            println!("return_value={}", outcome.return_value);
            println!("import_hits={}", outcome.import_hits);
            ExitCode::SUCCESS
        }
        Err(error) => {
            eprintln!("error: {}: {error}", path.display());
            ExitCode::FAILURE
        }
    }
}

pub(crate) fn run_game(path: &Path, import_limit: usize, timeout_seconds: u32) -> ExitCode {
    #[cfg(windows)]
    {
        run_game_native(path, import_limit, timeout_seconds)
    }
    #[cfg(target_os = "macos")]
    {
        run_game_native(path, import_limit, timeout_seconds)
    }
    #[cfg(not(any(target_os = "macos", windows)))]
    {
        let _ = (path, import_limit, timeout_seconds);
        eprintln!("error: no native graphics presenter is available on this host");
        ExitCode::FAILURE
    }
}

#[cfg(any(target_os = "macos", windows))]
fn run_game_native(path: &Path, import_limit: usize, timeout_seconds: u32) -> ExitCode {
    remu_gpu::begin_startup_timeline();
    let (linked, module_report) = match load_process_image(path) {
        Ok(process) => process,
        Err(error) => {
            eprintln!("error: {error}");
            return ExitCode::FAILURE;
        }
    };
    remu_gpu::record_startup_milestone(remu_gpu::StartupMilestone::ProcessImageLinked);
    let image_name = module_report
        .entry_path
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or("eboot.bin")
        .to_owned();
    let game_root = module_report.app_root;
    // Preserve command-stream order with bounded memory. If Metal is compiling a pipeline, the
    // queue eventually applies backpressure to the guest instead of dropping work or growing
    // without limit.
    let (sender, receiver) = remu_gpu::gpu_event_channel();
    remu_gpu::install_event_sink(sender.clone());
    remu_gpu::record_startup_milestone(remu_gpu::StartupMilestone::EventSinkInstalled);
    let guest = std::thread::spawn(move || {
        let started = Instant::now();
        let result = linked.probe_entry(&image_name, &game_root, import_limit, timeout_seconds);
        let active_threads = remu_runtime::active_guest_thread_count();
        if active_threads != 0 {
            eprintln!(
                "[REMu] guest entry returned with {active_threads} worker thread(s); keeping the runtime alive"
            );
        }
        while remu_runtime::active_guest_thread_count() != 0
            && (timeout_seconds == 0
                || started.elapsed() < Duration::from_secs(u64::from(timeout_seconds)))
        {
            std::thread::sleep(Duration::from_millis(10));
        }
        eprintln!(
            "[REMu] guest runtime stopped; requesting presenter shutdown (entry_result={})",
            if result.is_ok() { "returned" } else { "failed" }
        );
        let _ = sender.send(remu_gpu::GpuEvent::Shutdown);
        result
    });

    #[cfg(target_os = "macos")]
    let presenter_result = remu_gpu_metal::run_presenter(receiver);
    #[cfg(windows)]
    let presenter_result = remu_gpu_dx12::run_presenter(receiver);
    remu_gpu::clear_event_sink();
    remu_runtime::report_shader_opcode_coverage();
    if let Err(error) = presenter_result {
        eprintln!("error: {error}");
        return ExitCode::FAILURE;
    }
    if !guest.is_finished() {
        eprintln!("[REMu] presenter closed while guest threads were active; ending this REmu run");
        return ExitCode::SUCCESS;
    }
    match guest.join() {
        Ok(Ok(outcome)) => {
            eprintln!(
                "[REMu] guest returned {} after {} imports",
                outcome.return_value, outcome.import_hits
            );
            ExitCode::SUCCESS
        }
        Ok(Err(error)) => {
            eprintln!("error: {}: {error}", path.display());
            ExitCode::FAILURE
        }
        Err(_) => {
            eprintln!("error: the guest execution thread panicked");
            ExitCode::FAILURE
        }
    }
}

fn load_process_image(
    executable_path: &Path,
) -> Result<(remu_runtime::LinkedImage, remu_runtime::ModuleLinkReport), String> {
    let (image, report) = remu_runtime::LinkedImage::link_path(executable_path)
        .map_err(|error| format!("{}: {error}", executable_path.display()))?;
    for module in &report.loaded_modules {
        eprintln!("[REMu][MODULE] loaded {}", module.display());
    }
    eprintln!(
        "[REMu][MODULE] dependency graph: bundled={} system/HLE={} app_root={}",
        report.loaded_modules.len(),
        report.unresolved_dependencies.len(),
        report.app_root.display()
    );
    Ok((image, report))
}
