// Licensed under the Apache License, Version 2.0.

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
            if let Some(elf) = info.elf {
                println!("elf.os_abi={}", elf.os_abi);
                println!("elf.abi_version={}", elf.abi_version);
                println!("elf.object_type=0x{:04x}", elf.object_type);
                println!("elf.machine=0x{:04x}", elf.machine);
                println!("elf.entry={}", elf.entry);
                println!("elf.program_headers={}", elf.program_header_count);
                println!(
                    "elf.program_header_offset=0x{:x}",
                    elf.program_header_offset
                );
                println!("elf.section_headers={}", elf.section_header_count);
                println!(
                    "elf.section_header_offset=0x{:x}",
                    elf.section_header_offset
                );
            }
            ExitCode::SUCCESS
        }
        Err(error) => {
            eprintln!("error: {}: {error}", path.display());
            ExitCode::FAILURE
        }
    }
}
