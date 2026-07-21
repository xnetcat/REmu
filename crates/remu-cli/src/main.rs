// SPDX-License-Identifier: GPL-2.0-or-later

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::process::ExitCode;

use clap::{Parser, Subcommand};
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
    /// Execute a bounded native probe and print the first imported calls.
    Probe {
        path: PathBuf,
        #[arg(long, default_value_t = 32)]
        import_limit: usize,
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
        Command::Probe { path, import_limit } => probe(&path, import_limit),
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
                    println!("dynamic.symbols={}", dynamic.symbols.len());
                    println!("dynamic.relocations={}", dynamic.relocations.len());
                    println!("dynamic.plt_relocations={}", dynamic.jump_relocations.len());
                    println!("dynamic.imports={}", dynamic.imports.len());
                    for library in dynamic.needed_libraries {
                        println!("dynamic.needed={library}");
                    }
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
    let executable = match remu_loader::load_path(path) {
        Ok(executable) => executable,
        Err(error) => {
            eprintln!("error: {}: {error}", path.display());
            return ExitCode::FAILURE;
        }
    };
    let linked = match remu_runtime::LinkedImage::link_main(&executable) {
        Ok(linked) => linked,
        Err(error) => {
            eprintln!("error: {}: {error}", path.display());
            return ExitCode::FAILURE;
        }
    };
    println!("path={}", path.display());
    println!("image_base={}", linked.memory.image_base());
    println!("entry={}", linked.memory.entry_address());
    println!("import_stubs={}", linked.import_stubs.len());
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

fn probe(path: &Path, import_limit: usize) -> ExitCode {
    let executable = match remu_loader::load_path(path) {
        Ok(executable) => executable,
        Err(error) => {
            eprintln!("error: {}: {error}", path.display());
            return ExitCode::FAILURE;
        }
    };
    let linked = match remu_runtime::LinkedImage::link_main(&executable) {
        Ok(linked) => linked,
        Err(error) => {
            eprintln!("error: {}: {error}", path.display());
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
    let image_name = path
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or("eboot.bin");
    let game_root = path.parent().unwrap_or_else(|| Path::new("."));
    match linked.probe_entry(image_name, game_root, import_limit) {
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
