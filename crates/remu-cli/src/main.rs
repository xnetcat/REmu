// SPDX-License-Identifier: GPL-2.0-or-later

use std::path::PathBuf;
use std::process::ExitCode;

use clap::{Parser, Subcommand};
#[cfg(target_os = "macos")]
use remu_gpu::GpuProbe;
#[cfg(target_os = "macos")]
use remu_gpu_metal::MetalProbe;

mod analyze;
mod execute;

use analyze::{disassemble, imports, inspect};
use execute::{link, map, probe, run_game};

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
    // All stderr diagnostics become asynchronous for the whole process: authoring threads pay
    // for a pipe copy, one pump thread pays for the file. REMU_SYNC_STDERR=1 opts out.
    remu_memory::install_async_stderr();
    let code = main_command();
    // Exit-path reports ([REMu][REPORT]) must land in the log before the process ends.
    remu_memory::flush_async_stderr();
    code
}

fn main_command() -> ExitCode {
    match Arguments::parse().command {
        Command::Inspect { path } => inspect(&path),
        Command::HostInfo => {
            let info = remu_platform::host_info();
            println!("os={}", info.operating_system);
            println!("architecture={}", info.architecture);
            println!("process_translation={}", info.process_translation);
            ExitCode::SUCCESS
        }
        Command::GpuInfo => gpu_info(),
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

#[cfg(target_os = "macos")]
fn gpu_info() -> ExitCode {
    match MetalProbe::probe_default() {
        Ok(adapter) => {
            println!("api={}", adapter.api);
            println!("adapter={}", adapter.name);
            ExitCode::SUCCESS
        }
        Err(error) => {
            eprintln!("error: {error}");
            ExitCode::FAILURE
        }
    }
}

#[cfg(not(target_os = "macos"))]
fn gpu_info() -> ExitCode {
    eprintln!("error: gpu-info requires the macOS Metal backend");
    ExitCode::FAILURE
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
