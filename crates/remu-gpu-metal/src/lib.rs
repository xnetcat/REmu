// SPDX-License-Identifier: GPL-2.0-or-later

use remu_gpu::{GpuAdapter, GpuProbe, GpuProbeError, GraphicsApi};

#[cfg(target_os = "macos")]
mod presenter;

#[cfg(target_os = "macos")]
pub use presenter::run_presenter;

#[cfg(not(target_os = "macos"))]
pub fn run_presenter(
    _receiver: std::sync::mpsc::Receiver<remu_gpu::GpuEvent>,
) -> Result<(), String> {
    Err("the Metal presenter is available only on macOS".to_owned())
}

pub struct MetalProbe;

#[cfg(target_os = "macos")]
impl GpuProbe for MetalProbe {
    fn probe_default() -> Result<GpuAdapter, GpuProbeError> {
        let device =
            metal::Device::system_default().ok_or(GpuProbeError::NoDevice(GraphicsApi::Metal))?;

        Ok(GpuAdapter {
            api: GraphicsApi::Metal,
            name: device.name().to_owned(),
        })
    }
}

#[cfg(not(target_os = "macos"))]
impl GpuProbe for MetalProbe {
    fn probe_default() -> Result<GpuAdapter, GpuProbeError> {
        Err(GpuProbeError::Unsupported(GraphicsApi::Metal))
    }
}
