// SPDX-License-Identifier: GPL-2.0-or-later

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
