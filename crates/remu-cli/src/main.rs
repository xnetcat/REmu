// SPDX-License-Identifier: GPL-2.0-or-later

use std::path::PathBuf;
use std::process::ExitCode;

use clap::{Parser, Subcommand};
#[cfg(target_os = "macos")]
use remu_gpu::GpuProbe;
#[cfg(windows)]
use remu_gpu_dx12::{probe_default_capabilities, run_headless_self_test, run_pipeline_self_test};
#[cfg(target_os = "macos")]
use remu_gpu_metal::MetalProbe;

mod analyze;
mod execute;

use analyze::{disassemble, imports, inspect};
use execute::{link, map, probe, run_game};

#[derive(Debug, Parser)]
#[command(name = "remu", version, about = "REmu diagnostics and game launcher")]
struct Arguments {
    /// Enable the Direct3D 12 SDK debug layer (requires Windows Graphics Tools).
    #[arg(long, global = true)]
    d3d12_debug: bool,
    /// Enable slow GPU-based D3D12 validation in addition to the SDK debug layer.
    #[arg(long, global = true)]
    d3d12_gpu_validation: bool,
    /// Enable DRED auto-breadcrumb, breadcrumb-context, and page-fault capture.
    #[arg(long, global = true)]
    d3d12_dred: bool,
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
    /// Submit and validate a real Direct3D 12 upload/copy/readback workload.
    GpuSelfTest,
    /// Open a short-lived Direct3D 12 swap-chain window and present a diagnostic frame.
    GpuPresentTest,
    /// Open the default host audio output and print its negotiated format.
    AudioInfo,
    /// Report connected Windows XInput controllers and optionally pulse their motors.
    InputInfo {
        /// Run a short, low-strength two-motor vibration test on each connected controller.
        #[arg(long)]
        rumble_test: bool,
    },
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
    /// Launch a game with the native graphics presentation window.
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
    let arguments = Arguments::parse();
    #[cfg(windows)]
    if let Err(error) = remu_gpu_dx12::set_diagnostic_options(
        arguments.d3d12_debug,
        arguments.d3d12_gpu_validation,
        arguments.d3d12_dred,
    ) {
        eprintln!("error: {error}");
        return ExitCode::FAILURE;
    }
    match arguments.command {
        Command::Inspect { path } => inspect(&path),
        Command::HostInfo => {
            let info = remu_platform::host_info();
            println!("os={}", info.operating_system);
            println!("architecture={}", info.architecture);
            println!("process_translation={}", info.process_translation);
            ExitCode::SUCCESS
        }
        Command::GpuInfo => gpu_info(),
        Command::GpuSelfTest => gpu_self_test(),
        Command::GpuPresentTest => gpu_present_test(),
        Command::AudioInfo => audio_info(),
        Command::InputInfo { rumble_test } => input_info(rumble_test),
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

#[cfg(windows)]
fn gpu_self_test() -> ExitCode {
    match run_headless_self_test() {
        Ok(report) => {
            println!("api=Direct3D 12");
            println!("adapter={}", report.adapter);
            println!("copied_bytes={}", report.copied_bytes);
            println!("checksum=0x{:016x}", report.checksum);
            match run_pipeline_self_test() {
                Ok(pipeline) => {
                    println!("shader_model={}", pipeline.shader_model);
                    println!("colored_pixels={}", pipeline.colored_pixels);
                    println!("computed_values={}", pipeline.computed_values);
                    println!("guest_computed_values={}", pipeline.guest_computed_values);
                    println!("result=pass");
                    ExitCode::SUCCESS
                }
                Err(error) => {
                    eprintln!("error: {error}");
                    ExitCode::FAILURE
                }
            }
        }
        Err(error) => {
            eprintln!("error: {error}");
            ExitCode::FAILURE
        }
    }
}

#[cfg(windows)]
fn gpu_present_test() -> ExitCode {
    let width = 1280_u32;
    let height = 720_u32;
    let mut rgba = Vec::with_capacity((width * height * 4) as usize);
    for y in 0..height {
        for x in 0..width {
            rgba.extend_from_slice(&[
                u8::try_from(x * 255 / width).expect("gradient red fits u8"),
                u8::try_from(y * 255 / height).expect("gradient green fits u8"),
                96,
                255,
            ]);
        }
    }
    let Some(image) = remu_gpu::GpuHostImage::new(width, height, rgba) else {
        eprintln!("error: could not build the presentation diagnostic frame");
        return ExitCode::FAILURE;
    };
    let (sender, receiver) = remu_gpu::gpu_event_channel();
    remu_gpu::install_event_sink(sender);
    let events = std::thread::spawn(move || {
        let _ = remu_gpu::emit_event(remu_gpu::GpuEvent::ShowHostSplash(image));
        std::thread::sleep(std::time::Duration::from_secs(2));
        let _ = remu_gpu::emit_event(remu_gpu::GpuEvent::Shutdown);
    });
    let result = remu_gpu_dx12::run_presenter(receiver);
    remu_gpu::clear_event_sink();
    let _ = events.join();
    match result {
        Ok(()) => {
            println!("result=pass");
            ExitCode::SUCCESS
        }
        Err(error) => {
            eprintln!("error: {error}");
            ExitCode::FAILURE
        }
    }
}

#[cfg(not(windows))]
fn gpu_present_test() -> ExitCode {
    eprintln!("error: the Direct3D 12 presentation test is available only on Windows");
    ExitCode::FAILURE
}

#[cfg(not(windows))]
fn gpu_self_test() -> ExitCode {
    eprintln!("error: the Direct3D 12 self-test is available only on Windows");
    ExitCode::FAILURE
}

fn audio_info() -> ExitCode {
    match remu_runtime::probe_host_audio() {
        Ok(info) => {
            println!("sample_rate={}", info.sample_rate);
            println!("channels={}", info.channels);
            ExitCode::SUCCESS
        }
        Err(error) => {
            eprintln!("error: {error}");
            ExitCode::FAILURE
        }
    }
}

#[cfg(windows)]
fn input_info(rumble_test: bool) -> ExitCode {
    let controllers = remu_gpu_dx12::probe_xinput_controllers();
    println!("connected_controllers={}", controllers.len());
    for controller in &controllers {
        println!(
            "controller[{}]=packet:{} buttons:0x{:04x} triggers:{}/{} left:{}/{} right:{}/{}",
            controller.user_index,
            controller.packet_number,
            controller.buttons,
            controller.left_trigger,
            controller.right_trigger,
            controller.left_x,
            controller.left_y,
            controller.right_x,
            controller.right_y,
        );
        if rumble_test {
            match remu_gpu_dx12::test_xinput_vibration(controller.user_index) {
                Ok(()) => println!("controller[{}].rumble=pass", controller.user_index),
                Err(error) => {
                    eprintln!(
                        "controller[{}].rumble=failed: {error}",
                        controller.user_index
                    );
                    return ExitCode::FAILURE;
                }
            }
        }
    }
    ExitCode::SUCCESS
}

#[cfg(not(windows))]
fn input_info(_rumble_test: bool) -> ExitCode {
    eprintln!("error: XInput diagnostics are available only on Windows");
    ExitCode::FAILURE
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
#[cfg(not(windows))]
fn gpu_info() -> ExitCode {
    eprintln!("error: no native GPU backend is available on this host");
    ExitCode::FAILURE
}

#[cfg(windows)]
fn gpu_info() -> ExitCode {
    match probe_default_capabilities() {
        Ok(adapter) => {
            println!("api=Direct3D 12");
            println!("adapter={}", adapter.name);
            println!("vendor_id=0x{:04x}", adapter.vendor_id);
            println!("device_id=0x{:04x}", adapter.device_id);
            println!("adapter_luid=0x{:016x}", adapter.adapter_luid);
            println!(
                "dedicated_video_memory_mib={}",
                adapter.dedicated_video_memory / (1024 * 1024)
            );
            println!(
                "local_memory_budget_mib={}",
                adapter.local_memory.budget / (1024 * 1024)
            );
            println!(
                "local_memory_current_usage_mib={}",
                adapter.local_memory.current_usage / (1024 * 1024)
            );
            println!(
                "local_memory_available_for_reservation_mib={}",
                adapter.local_memory.available_for_reservation / (1024 * 1024)
            );
            println!(
                "local_memory_current_reservation_mib={}",
                adapter.local_memory.current_reservation / (1024 * 1024)
            );
            println!(
                "non_local_memory_budget_mib={}",
                adapter.non_local_memory.budget / (1024 * 1024)
            );
            println!(
                "non_local_memory_current_usage_mib={}",
                adapter.non_local_memory.current_usage / (1024 * 1024)
            );
            println!("feature_level={}", adapter.feature_level_name());
            println!("shader_model={}", adapter.shader_model_name());
            println!("resource_binding_tier={}", adapter.resource_binding_tier);
            println!("resource_heap_tier={}", adapter.resource_heap_tier);
            println!("wave_ops={}", adapter.wave_ops);
            println!("int64_shader_ops={}", adapter.int64_shader_ops);
            println!(
                "wave_lane_range={}..={}",
                adapter.wave_lane_count_min, adapter.wave_lane_count_max
            );
            println!("mesh_shader_tier={}", adapter.mesh_shader_tier);
            println!("uma={}", adapter.uma);
            println!("cache_coherent_uma={}", adapter.cache_coherent_uma);
            println!("enhanced_barriers={}", adapter.enhanced_barriers);
            println!("writable_msaa_textures={}", adapter.writable_msaa_textures);
            println!(
                "independent_front_back_stencil_ref_mask={}",
                adapter.independent_front_back_stencil_ref_mask
            );
            println!("triangle_fan={}", adapter.triangle_fan);
            println!(
                "dynamic_index_buffer_strip_cut={}",
                adapter.dynamic_index_buffer_strip_cut
            );
            println!("dynamic_depth_bias={}", adapter.dynamic_depth_bias);
            println!("supported={}", adapter.is_supported());
            ExitCode::SUCCESS
        }
        Err(error) => {
            eprintln!("error: {error}");
            ExitCode::FAILURE
        }
    }
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
