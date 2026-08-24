// SPDX-License-Identifier: GPL-2.0-or-later

//! Subcommands that materialize guest memory and run guest code.

use std::path::Path;
use std::process::ExitCode;
#[cfg(target_os = "macos")]
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
    for (dispatch_index, stub) in linked.import_stubs.iter().enumerate() {
        println!(
            "import_stub[{dispatch_index}]=address:{} nid:{}",
            stub.address, stub.nid
        );
    }
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
    spawn_timed_guest_memory_dump();
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
    #[cfg(not(target_os = "macos"))]
    {
        let _ = (path, import_limit, timeout_seconds);
        eprintln!("error: run requires the macOS Metal presenter");
        return ExitCode::FAILURE;
    }
    #[cfg(target_os = "macos")]
    {
        run_game_metal(path, import_limit, timeout_seconds)
    }
}

#[cfg(target_os = "macos")]
fn run_game_metal(path: &Path, import_limit: usize, timeout_seconds: u32) -> ExitCode {
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
    spawn_timed_guest_memory_dump();
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

    let presenter_result = remu_gpu_metal::run_presenter(receiver);
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

/// `REMU_DUMP_MEMORY_AT_SECONDS=N`: snapshot every shared guest mapping to
/// `$REMU_CAPTURE_DIR/memdump/` N seconds after guest entry, for offline analysis of live
/// structures (descriptor heaps, staged tables, render-target RAM) at a chosen moment — e.g.
/// parked at a title menu. Reads go through the checked shared-backing path, so a mapping
/// retired mid-dump skips instead of faulting.
fn spawn_timed_guest_memory_dump() {
    let Some(seconds) = std::env::var("REMU_DUMP_MEMORY_AT_SECONDS")
        .ok()
        .and_then(|value| value.parse::<u64>().ok())
    else {
        return;
    };
    let directory = std::env::var_os("REMU_CAPTURE_DIR")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(std::env::temp_dir)
        .join("memdump");
    let _ = std::thread::Builder::new()
        .name("remu-memdump".into())
        .spawn(move || {
            std::thread::sleep(std::time::Duration::from_secs(seconds));
            if let Err(error) = std::fs::create_dir_all(&directory) {
                eprintln!(
                    "[REMu][MEMDUMP] cannot create {}: {error}",
                    directory.display()
                );
                return;
            }
            let table = remu_memory::guest_shared_mapping_table();
            let mut index = std::fs::File::create(directory.join("mappings.tsv")).ok();
            let mut dumped = 0_u64;
            for (address, length, backing_id, backing_offset) in &table {
                use std::io::Write as _;
                if let Some(index) = index.as_mut() {
                    let _ = writeln!(
                        index,
                        "{address:#x}\t{length:#x}\t{backing_id}\t{backing_offset:#x}"
                    );
                }
                let path = directory.join(format!("mem-{address:016x}.bin"));
                let Ok(mut file) = std::fs::File::create(&path) else {
                    continue;
                };
                let mut chunk = vec![0_u8; 16 * 1024 * 1024];
                let mut cursor = 0_u64;
                while cursor < *length {
                    let step = chunk.len().min((*length - cursor) as usize);
                    if !remu_memory::read_guest_shared_backing(address + cursor, &mut chunk[..step])
                    {
                        break;
                    }
                    if file.write_all(&chunk[..step]).is_err() {
                        break;
                    }
                    cursor += step as u64;
                    dumped += step as u64;
                }
            }
            eprintln!(
                "[REMu][MEMDUMP] wrote {} mappings, {} MiB, to {}",
                table.len(),
                dumped / (1024 * 1024),
                directory.display()
            );
        });
}
