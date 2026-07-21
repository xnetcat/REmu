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
            }
            ExitCode::SUCCESS
        }
        Err(error) => {
            eprintln!("error: {}: {error}", path.display());
            ExitCode::FAILURE
        }
    }
}
