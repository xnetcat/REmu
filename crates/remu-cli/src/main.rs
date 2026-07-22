// SPDX-License-Identifier: GPL-2.0-or-later

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::time::{Duration, Instant};

use clap::{Parser, Subcommand};
use iced_x86::{Decoder, DecoderOptions, Formatter, IntelFormatter};
use remu_gpu::GpuProbe;
use remu_gpu_metal::MetalProbe;

#[derive(Debug, Parser)]
#[command(name = "remu", version, about = "REmu diagnostics and game launcher")]
struct Arguments {
    #[command(subcommand)]
    command: Command,
}

#[derive(Debug, Subcommand)]
enum Command {
    /// Inspect an executable without loading or running it.
    Inspect { path: PathBuf },
    /// Print the host architecture and translation environment.
    HostInfo,
    /// Probe the native graphics backend selected for this host.
    GpuInfo,
    /// List module dependencies and imported symbols without executing the game.
    Imports { path: PathBuf },
    /// Materialize load segments at the PS5 guest base without executing code.
    Map { path: PathBuf },
    /// Map and relocate an executable, creating import trap stubs without executing it.
    Link { path: PathBuf },
    /// Disassemble bytes from a materialized guest load segment.
    Disassemble {
        path: PathBuf,
        /// Guest virtual address or mapped main-image address (hex accepts a 0x prefix).
        #[arg(value_parser = parse_integer)]
        address: u64,
        /// Maximum number of bytes to decode.
        #[arg(long, default_value_t = 128)]
        length: usize,
    },
    /// Execute a bounded native probe and print the first imported calls.
    Probe {
        path: PathBuf,
        #[arg(long, default_value_t = 32)]
        import_limit: usize,
        /// Stop the process after this many seconds; zero disables the timer.
        #[arg(long, default_value_t = 60)]
        timeout_seconds: u32,
    },
    /// Launch a game with the native Metal presentation window.
    Run {
        path: PathBuf,
        #[arg(long, default_value_t = 0)]
        import_limit: usize,
        /// Stop the process after this many seconds; zero disables the timer.
        #[arg(long, default_value_t = 0)]
        timeout_seconds: u32,
    },
}

fn main() -> ExitCode {
    match Arguments::parse().command {
        Command::Inspect { path } => inspect(&path),
        Command::HostInfo => {
            let info = remu_platform::host_info();
            println!("os={}", info.operating_system);
            println!("architecture={}", info.architecture);
            println!("process_translation={}", info.process_translation);
            ExitCode::SUCCESS
        }
        Command::GpuInfo => match MetalProbe::probe_default() {
            Ok(adapter) => {
                println!("api={}", adapter.api);
                println!("adapter={}", adapter.name);
                ExitCode::SUCCESS
            }
            Err(error) => {
                eprintln!("error: {error}");
                ExitCode::FAILURE
            }
        },
        Command::Imports { path } => imports(&path),
        Command::Map { path } => map(&path),
        Command::Link { path } => link(&path),
        Command::Disassemble {
            path,
            address,
            length,
        } => disassemble(&path, address, length),
        Command::Probe {
            path,
            import_limit,
            timeout_seconds,
        } => probe(&path, import_limit, timeout_seconds),
        Command::Run {
            path,
            import_limit,
            timeout_seconds,
        } => run_game(&path, import_limit, timeout_seconds),
    }
}

fn inspect(path: &Path) -> ExitCode {
    match remu_loader::inspect_path(path) {
        Ok(info) => {
            println!("path={}", path.display());
            println!("kind={}", info.kind);
            println!("file_size=0x{:x}", info.file_size);
            if let Some(container) = info.container {
                let header = container.header;
                println!("container.version={}", header.version);
                println!("container.mode={}", header.mode);
                println!("container.endian={}", header.endian);
                println!("container.attributes=0x{:02x}", header.attributes);
                println!("container.key_type=0x{:08x}", header.key_type);
                println!("container.header_size=0x{:x}", header.header_size);
                println!("container.metadata_size=0x{:x}", header.metadata_size);
                println!(
                    "container.declared_file_size=0x{:x}",
                    header.declared_file_size
                );
                println!("container.flags=0x{:04x}", header.flags);
                println!(
                    "container.embedded_elf_offset=0x{:x}",
                    header.embedded_elf_offset
                );
                for (index, segment) in container.segments.iter().enumerate() {
                    println!(
                        "container.segment[{index}]=program:{} properties:0x{:x} offset:0x{:x} stored:0x{:x} expanded:0x{:x} encrypted:{} blocks:{}",
                        segment.properties.program_index(),
                        segment.properties.bits(),
                        segment.file_offset,
                        segment.stored_size,
                        segment.expanded_size,
                        segment.properties.is_encrypted(),
                        segment.properties.has_blocks(),
                    );
                }
            }
            if let Some(elf) = info.elf {
                let header = elf.header;
                println!("elf.file_offset=0x{:x}", elf.file_offset);
                println!("elf.os_abi={}", header.os_abi);
                println!("elf.abi_version={}", header.abi_version);
                println!("elf.object_type=0x{:04x}", header.object_type);
                println!("elf.machine=0x{:04x}", header.machine);
                println!("elf.entry={}", header.entry);
                println!("elf.program_headers={}", header.program_header_count);
                println!(
                    "elf.program_header_offset=0x{:x}",
                    header.program_header_offset
                );
                println!("elf.section_headers={}", header.section_header_count);
                println!(
                    "elf.section_header_offset=0x{:x}",
                    header.section_header_offset
                );
                for (index, program) in elf.program_headers.iter().enumerate() {
                    println!(
                        "elf.program[{index}]=type:0x{:08x} flags:0x{:x} offset:0x{:x} vaddr:{} filesz:0x{:x} memsz:0x{:x} align:0x{:x}",
                        program.program_type,
                        program.flags,
                        program.file_offset,
                        program.virtual_address,
                        program.file_size,
                        program.memory_size,
                        program.alignment,
                    );
                }
                if let Some(dynamic) = elf.dynamic {
                    print_dynamic_info(dynamic);
                }
            }
            ExitCode::SUCCESS
        }
        Err(error) => {
            eprintln!("error: {}: {error}", path.display());
            ExitCode::FAILURE
        }
    }
}

fn print_dynamic_info(dynamic: remu_loader::DynamicInfo) {
    println!("dynamic.entries={}", dynamic.entries.len());
    println!(
        "dynamic.string_table=0x{:x}:0x{:x}",
        dynamic.string_table_address, dynamic.string_table_size
    );
    println!(
        "dynamic.symbol_table=0x{:x}:0x{:x}",
        dynamic.symbol_table_address, dynamic.symbol_table_size
    );
    println!(
        "dynamic.rela=0x{:x}:0x{:x}",
        dynamic.rela_address, dynamic.rela_size
    );
    println!(
        "dynamic.jump_relocations=0x{:x}:0x{:x}",
        dynamic.jump_relocation_address, dynamic.jump_relocation_size
    );
    println!("dynamic.init=0x{:x}", dynamic.init_address);
    println!(
        "dynamic.init_array=0x{:x}:0x{:x}",
        dynamic.init_array_address, dynamic.init_array_size
    );
    println!(
        "dynamic.preinit_array=0x{:x}:0x{:x}",
        dynamic.preinit_array_address, dynamic.preinit_array_size
    );
    println!("dynamic.symbols={}", dynamic.symbols.len());
    println!("dynamic.relocations={}", dynamic.relocations.len());
    println!("dynamic.plt_relocations={}", dynamic.jump_relocations.len());
    println!("dynamic.imports={}", dynamic.imports.len());
    for library in dynamic.needed_libraries {
        println!("dynamic.needed={library}");
    }
}

fn imports(path: &Path) -> ExitCode {
    let info = match remu_loader::inspect_path(path) {
        Ok(info) => info,
        Err(error) => {
            eprintln!("error: {}: {error}", path.display());
            return ExitCode::FAILURE;
        }
    };
    let Some(dynamic) = info.elf.and_then(|elf| elf.dynamic) else {
        eprintln!("error: {} has no dynamic-linker metadata", path.display());
        return ExitCode::FAILURE;
    };

    println!("path={}", path.display());
    println!("dependencies={}", dynamic.needed_libraries.len());
    for library in dynamic.needed_libraries {
        println!("dependency={library}");
    }
    println!("symbols={}", dynamic.symbols.len());
    println!(
        "relocations={}",
        dynamic.relocations.len() + dynamic.jump_relocations.len()
    );
    let mut relocation_types = BTreeMap::<u32, usize>::new();
    for relocation in dynamic.relocations.iter().chain(&dynamic.jump_relocations) {
        *relocation_types
            .entry(relocation.relocation_type)
            .or_default() += 1;
    }
    for (relocation_type, count) in relocation_types {
        println!("relocation_type[{relocation_type}]={count}");
    }
    println!("plt_entries={}", dynamic.jump_relocations.len());
    for (index, relocation) in dynamic.jump_relocations.iter().enumerate() {
        let symbol_name = usize::try_from(relocation.symbol_index)
            .ok()
            .and_then(|index| dynamic.symbols.get(index))
            .map_or("<missing>", |symbol| symbol.name.as_str());
        let nid = symbol_name
            .split_once('#')
            .map_or(symbol_name, |(nid, _)| nid);
        let resolved_name = remu_runtime::known_nid_name(nid).unwrap_or("unknown");
        println!(
            "plt[{index}]=offset:{} symbol:{} nid:{} resolved:{} name:{}",
            relocation.offset, relocation.symbol_index, nid, resolved_name, symbol_name
        );
    }
    let exports = dynamic
        .symbols
        .iter()
        .enumerate()
        .filter(|(_, symbol)| !symbol.is_undefined())
        .collect::<Vec<_>>();
    println!("exports={}", exports.len());
    for (index, symbol) in exports {
        println!(
            "export={} symbol={} value:{} size:0x{:x}",
            symbol.name, index, symbol.value, symbol.size
        );
    }
    println!("imports={}", dynamic.imports.len());
    for import in dynamic.imports {
        let relocation_types = import
            .relocation_types
            .iter()
            .map(u32::to_string)
            .collect::<Vec<_>>()
            .join(",");
        println!(
            "import={} symbol={} weak={} relocations={} types={}",
            import.name,
            import.symbol_index,
            import.weak,
            import.relocation_count,
            relocation_types
        );
    }
    ExitCode::SUCCESS
}

fn parse_integer(value: &str) -> Result<u64, String> {
    let compact = value.replace('_', "");
    if let Some(hex) = compact
        .strip_prefix("0x")
        .or_else(|| compact.strip_prefix("0X"))
    {
        u64::from_str_radix(hex, 16).map_err(|error| error.to_string())
    } else {
        compact.parse::<u64>().map_err(|error| error.to_string())
    }
}

fn disassemble(path: &Path, address: u64, length: usize) -> ExitCode {
    if length == 0 || length > 4096 {
        eprintln!("error: disassembly length must be between 1 and 4096 bytes");
        return ExitCode::FAILURE;
    }
    let executable = match remu_loader::load_path(path) {
        Ok(executable) => executable,
        Err(error) => {
            eprintln!("error: {}: {error}", path.display());
            return ExitCode::FAILURE;
        }
    };
    let image_base = remu_memory::PS5_MAIN_IMAGE_BASE.get();
    let virtual_address = address.checked_sub(image_base).unwrap_or(address);
    let Some(segment) = executable.load_segments.iter().find(|segment| {
        let start = segment.virtual_address.get();
        let end = start.saturating_add(u64::try_from(segment.data.len()).unwrap_or(u64::MAX));
        virtual_address >= start && virtual_address < end
    }) else {
        eprintln!("error: address {address:#x} is not backed by a file load segment");
        return ExitCode::FAILURE;
    };
    let offset = usize::try_from(virtual_address - segment.virtual_address.get())
        .expect("the selected segment offset fits usize");
    let bytes = &segment.data[offset..segment.data.len().min(offset.saturating_add(length))];
    let mapped_address = image_base.saturating_add(virtual_address);
    let mut decoder = Decoder::with_ip(64, bytes, mapped_address, DecoderOptions::NONE);
    let mut formatter = IntelFormatter::new();
    let mut text = String::new();
    while decoder.can_decode() {
        let instruction = decoder.decode();
        text.clear();
        formatter.format(&instruction, &mut text);
        let byte_offset = usize::try_from(instruction.ip() - mapped_address)
            .expect("decoded byte offset fits usize");
        let instruction_bytes = &bytes[byte_offset..][..instruction.len()];
        let hex = instruction_bytes
            .iter()
            .map(|byte| format!("{byte:02x}"))
            .collect::<Vec<_>>()
            .join(" ");
        println!("{:#018x} {hex:<44} {text}", instruction.ip());
    }
    ExitCode::SUCCESS
}

fn map(path: &Path) -> ExitCode {
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

fn link(path: &Path) -> ExitCode {
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

fn probe(path: &Path, import_limit: usize, timeout_seconds: u32) -> ExitCode {
    let (linked, module_report) = match load_process_image(path) {
        Ok(process) => process,
        Err(error) => {
            eprintln!("error: {error}");
            return ExitCode::FAILURE;
        }
    };
    eprintln!(
        "[REMu][PROBE] entering {} at {} with {} import stubs, TLS {} ({} patched loads)",
        path.display(),
        linked.memory.entry_address(),
        linked.import_stubs.len(),
        linked.tls.thread_pointer,
        linked.tls.patched_loads,
    );
    let image_name = module_report
        .entry_path
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or("eboot.bin");
    match linked.probe_entry(
        image_name,
        &module_report.app_root,
        import_limit,
        timeout_seconds,
    ) {
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

fn run_game(path: &Path, import_limit: usize, timeout_seconds: u32) -> ExitCode {
    let (linked, module_report) = match load_process_image(path) {
        Ok(process) => process,
        Err(error) => {
            eprintln!("error: {error}");
            return ExitCode::FAILURE;
        }
    };
    let image_name = module_report
        .entry_path
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or("eboot.bin")
        .to_owned();
    let game_root = module_report.app_root;
    // Do not block the guest while it holds the import-dispatch lock. Explicit GPU ordering events
    // still serialize cache invalidation with preceding and following translated work.
    let (sender, receiver) = std::sync::mpsc::channel();
    remu_gpu::install_event_sink(sender.clone());
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
